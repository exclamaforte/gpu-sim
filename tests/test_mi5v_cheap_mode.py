"""Pin: cheap mode must not be a different check from the gate's full mode (gpu-sim-mi5v).

tests/test_append_only.py runs the retirement tests in cheap mode
(--no-ledger-replay) so ~46 minutes of ledger replay leaves the suite. That is
only sound if cheap mode gives the SAME verdict as full mode on the same tree:
"the suite tests mode B while the gate runs mode A" is the 0fr1/nfxi disease.

These tests build a MINI repo (no ledger to replay, so full mode is fast) and
compare (rc, FAIL lines) between the two modes on a passing fixture and on two
failing fixtures. The registered mutation makes cheap mode always pass, which
must make the failing-fixture comparisons fail.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_append_only.py"


def _mini_repo(td, delete_key=False, row_nodes=()):
    """Smallest tree the checker accepts: git repo + facts + one ledger row."""
    repo = Path(td) / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "tests" / "fs2y_ledger.d").mkdir(parents=True)
    (repo / "scripts" / "fs2y_mutations.d").mkdir(parents=True)
    # The checker resolves ROOT from its OWN path, so it must live in the
    # fixture: running the real one with cwd=<fixture> replayed the real ledger
    # (measured: 120s timeout before this copy existed).
    (repo / "scripts" / "check_append_only.py").write_text(CHECKER.read_text())
    # the grandfather check imports this helper from scripts/
    (repo / "scripts" / "fs2y_mutations.py").write_text(
        (ROOT / "scripts" / "fs2y_mutations.py").read_text())
    (repo / "scripts" / "resolve_trunk.py").write_text(
        (ROOT / "scripts" / "resolve_trunk.py").read_text())
    # gpu-sim-als2: check_append_only.py imports plant_verify from scripts/ to
    # decide whether a plant failed as a TEST OUTCOME (gpu-sim-vuoq), so this
    # module must sit beside the copy or the import dies before any row runs.
    (repo / "scripts" / "plant_verify.py").write_text(
        (ROOT / "scripts" / "plant_verify.py").read_text())
    # One real registry mutation, so a fixture row's mutation field resolves.
    for fn in ("append_only_guard.py", "append_only_guard.json"):
        (repo / "scripts" / "fs2y_mutations.d" / fn).write_text(
            (ROOT / "scripts" / "fs2y_mutations.d" / fn).read_text())
    facts = {"keep_me": {"v": 1}, "gone_me": {"v": 2}}
    (repo / "docs" / "model_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True,
                                    capture_output=True, text=True)
    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-qm", "base")
    # Move HEAD off main so the checker does not treat the fixture as the
    # integration tree (it refuses to run there without --force, and cheap mode
    # refuses --force by design -- the equivalence comparison needs both modes).
    git("checkout", "-q", "-b", "probe")
    (repo / "README").write_text("probe\n")
    git("add", "-A")
    git("commit", "-qm", "probe")
    if delete_key:
        # The deletion must be in the WORKING TREE against a base that still has
        # the key: committing the deletion would make base == head and the loss
        # invisible (the checker compares base_ref's file to the worktree's).
        del facts["gone_me"]
        (repo / "docs" / "model_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    # A real ledger row (and the module it names) so the resolve-only walk has
    # something to resolve.
    (repo / "tests" / "test_ok.py").write_text(
        "import pytest\n\n"
        "def test_a():\n    assert True\n\n"
        "class TestK:\n    def test_b(self):\n        assert True\n\n"
        # gpu-sim-mi5v: an id with a SPACE in it. `pytest --collect-only -q`
        # prints this node verbatim, and a parser that splits the line on ' '
        # truncates it to 'test_p[a' -- so cheap resolution reports a row
        # unresolvable that the per-row walk resolves. The parametrized id is
        # the fixture's whole purpose; do not 'simplify' it away.
        "@pytest.mark.parametrize('p', ['a b', 'c'])\n"
        "def test_p(p):\n    assert p\n")
    for i, node in enumerate(row_nodes):
        (repo / "tests" / "fs2y_ledger.d" / f"row_{i}.json").write_text(json.dumps({
            "name": f"row_{i}", "beads": ["gpu-sim-mi5v"], "state": "passing",
            "test": node, "bound_calls": 1,
            "mutation": "append_only_guard",
        }, indent=2) + "\n")
    git("add", "-A")
    git("commit", "-qm", "rows", "--allow-empty")
    return repo


def _run(repo, cheap):
    """cheap=True -> --no-ledger-replay; "resolve" -> --resolve-node-ids."""
    args = [sys.executable, str(repo / "scripts" / "check_append_only.py"),
            "--dirty-ok", "--base", "main"]
    if cheap == "resolve":
        args.append("--resolve-node-ids")
    elif cheap:
        args.append("--no-ledger-replay")
    r = subprocess.run(args, cwd=str(repo), capture_output=True, text=True, timeout=120)
    fails = tuple(l for l in ((r.stdout or "") + (r.stderr or "")).splitlines()
                  if l.startswith("FAIL:"))
    return r.returncode, fails


class TestCheapModeEquivalence(unittest.TestCase):
    def test_cheap_and_full_agree_on_a_passing_fixture(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / ".lab" / "tmp")) as td:
            repo = _mini_repo(td)
            self.assertEqual(_run(repo, cheap=True), _run(repo, cheap=False),
                             "cheap and full disagree on a passing fixture")

    def test_cheap_and_full_agree_on_a_failing_fixture(self):
        with tempfile.TemporaryDirectory(dir=str(ROOT / ".lab" / "tmp")) as td:
            repo = _mini_repo(td, delete_key=True)
            cheap, full = _run(repo, cheap=True), _run(repo, cheap=False)
            self.assertNotEqual(full[0], 0, "fixture did not fail in full mode")
            self.assertEqual(cheap, full,
                             "cheap mode does not reproduce the full-mode failure")

    def test_resolve_only_agrees_with_full_on_a_resolving_ledger(self):
        """Resolution verdicts are identical: batched collect vs the full walk."""
        with tempfile.TemporaryDirectory(dir=str(ROOT / ".lab" / "tmp")) as td:
            repo = _mini_repo(td, row_nodes=["tests/test_ok.py::test_a",
                                             "tests/test_ok.py::TestK::test_b"])
            full = _run(repo, cheap=False)
            res = _run(repo, "resolve")
            self.assertEqual(full[0], 0, full)
            self.assertEqual(res, full,
                             "resolve-only disagrees with the full walk on a "
                             "ledger whose nodes all resolve")

    def test_resolve_only_and_full_both_reject_an_unresolvable_node(self):
        """A row naming a missing test must fail BOTH modes -- the pin that the
        cheap resolution cannot be satisfied by simply walking fewer nodes."""
        with tempfile.TemporaryDirectory(dir=str(ROOT / ".lab" / "tmp")) as td:
            repo = _mini_repo(td, row_nodes=["tests/test_ok.py::does_not_exist"])
            full = _run(repo, cheap=False)
            res = _run(repo, "resolve")
            self.assertNotEqual(full[0], 0, "fixture did not fail in full mode")
            self.assertNotEqual(res[0], 0, "resolve-only passed an unresolvable node")
            self.assertTrue(any("does not resolve" in l for l in full[1]), full)
            self.assertTrue(any("does not resolve" in l for l in res[1]), res)

    def test_resolve_only_agrees_with_full_on_a_node_id_containing_a_space(self):
        """The case where the cheap mode silently became a DIFFERENT check.

        A parametrized node id can contain a space ('test_p[a b]'). The batched
        collect used to split each collected line on ' ', so resolve-only mode
        saw 'tests/test_ok.py::test_p[a' and failed a row that the per-row walk
        (node id passed as one argv element) resolved. Both modes must return
        the same verdict here, or "cheap == full" is only true for ids without
        spaces -- exactly the assumption that turned this file into a 300s
        budget failure in the first place.
        """
        with tempfile.TemporaryDirectory(dir=str(ROOT / ".lab" / "tmp")) as td:
            repo = _mini_repo(td, row_nodes=["tests/test_ok.py::test_p[a b]"])
            full = _run(repo, cheap=False)
            res = _run(repo, "resolve")
            self.assertEqual(full[0], 0, full)
            self.assertEqual(res, full,
                             "resolve-only disagrees with the full walk on a "
                             "node id containing a space")

    def test_cheap_mode_is_refused_with_force(self):
        r = subprocess.run([sys.executable, str(CHECKER), "--force", "--no-ledger-replay"],
                           cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("cannot be combined with --force", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
