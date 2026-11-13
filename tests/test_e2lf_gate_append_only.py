"""e2lf: the append-only stage runs in a disposable worktree (gpu-sim-e2lf).

Why the stage exists: the landing gate runs in the integration tree, where
HEAD == main by construction, and the append-only walk refuses to run there
(yy6o's guard) because a walk in the integration tree mutates the tree being
landed.  scripts/run_append_only_stage.sh runs the same walk in a throwaway
worktree of the same commit, so the check still runs -- and can still fail --
while the integration tree is never touched.

These tests drive that script directly, which is exactly what
scripts/gate.sh's "append-only records" stage runs.  Driving the script rather
than the whole gate keeps the test bounded and avoids a gate that invokes the
gate while still exercising the real code path, including cleanup.

The decisive assertion in the failing case is the walker's own message naming
the offending row -- `rc != 0` alone would also be satisfied by a missing file,
a syntax error or a timeout, which is the failure mode that let three
"plants" look green earlier in this repo.
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "scripts" / "run_append_only_stage.sh"
GATE = ROOT / "scripts" / "gate.sh"
ROW_REL = Path("tests/fs2y_ledger.d/ypdk_tally.json")


def _git(*args, cwd=ROOT):
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True, check=True)


def _run_stage(ref=None):
    cmd = ["bash", str(STAGE)]
    if ref:
        cmd.append(ref)
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=900)


def _appendonly_worktrees():
    """Registrations of the stage's own disposable worktree.

    Matched by exact path suffix `<tmp>/appendonly.<pid>`: the walker's own
    tests create worktrees INSIDE the disposable tree (`.../appendonly.N/.lab/
    tmp/zsd3-pro...`), and a substring match would report those as leftovers.
    The invariant is about registrations, not directories: a registration that
    survives makes every later `git worktree list` noisier and can collide with
    the next run's prune.
    """
    pattern = re.compile(r"/appendonly\.\d+$")
    return [line for line in _git("worktree", "list").stdout.splitlines()
            if pattern.search(line)]


def test_gate_runs_the_stage_script_not_an_inline_walk():
    """The gate must actually call the extracted stage (and keep it HARD)."""
    text = GATE.read_text()
    assert "run_append_only_stage.sh" in text, "gate.sh does not call the stage script"
    assert 'run_hard "append-only records"' in text, "append-only stage is not a HARD stage"


# NOTE: there is deliberately no "valid tip passes" test here. It would cost a
# second full walk (~110 s) and prove what every real gate run already proves.
# The failing case below is the one that can regress silently: it shows the
# stage REACHES the walker and that a violation is caught by name.


def test_stage_fails_on_induced_violation_and_leaves_no_worktree():
    tmp = Path(tempfile.mkdtemp(prefix="e2lf-stage-", dir=str(ROOT / ".lab" / "tmp")))
    tw = tmp / "tw"
    try:
        _git("worktree", "add", "--detach", str(tw), "HEAD")
        row = tw / ROW_REL
        data = json.loads(row.read_text())
        data["mutation"] = "n/a"
        row.write_text(json.dumps(data, indent=2) + "\n")
        _git("add", str(ROW_REL), cwd=tw)
        _git("-c", "user.email=test@example.invalid", "-c", "user.name=e2lf-test",
             "commit", "-qm", "e2lf induced violation (test)", cwd=tw)
        sha = _git("rev-parse", "HEAD", cwd=tw).stdout.strip()

        result = _run_stage(sha)
        out = result.stdout + result.stderr
        assert result.returncode != 0, f"stage passed on an induced violation:\n{out}"
        # the decisive assertion: the walker names the offending row
        assert "FAIL: ledger row 'ypdk_tally'" in out, out
        assert "no registered mutation function" in out, out
        assert _appendonly_worktrees() == [], _appendonly_worktrees()
    finally:
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(tw)],
                       capture_output=True, text=True)
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "prune"],
                       capture_output=True, text=True)
