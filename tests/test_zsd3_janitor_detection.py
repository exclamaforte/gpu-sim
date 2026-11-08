"""gpu-sim-zsd3: the tree janitor must DETECT residue, never repair it.

Why a subprocess: the thing under test is the autouse fixture's TEARDOWN, and no
test can honestly observe its own teardown (it has not happened yet). Each case
below therefore runs a real pytest on a tiny probe file and inspects the effect
on the repository afterwards.

The two properties that matter, both taken from the bug that motivated the bead:

  * work that was already uncommitted when the run started must still be there
    afterwards -- the old janitor reverted exactly this, which is how it reverted
    a real fix mid-run and made a test fail on content it had itself restored;
  * residue produced DURING a run must be REPORTED loudly and LEFT IN PLACE --
    the old janitor wrote HEAD content over it (and unlinked debris), so the
    failure was invisible in the diff and could be re-explained as flaky.

A repairer and a detector are indistinguishable while everything passes; these
assertions only have teeth because each probe pollutes a tracked file on purpose.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROBE_DIR = REPO / ".lab" / "tmp" / "zsd3_probe"

# A tracked file that real plant tests also use, so the probe exercises the same
# paths the fixture used to rewrite. Restored in a finally in every case.
TARGET_REL = "docs/model_facts.json"
TARGET = REPO / TARGET_REL


def _run_probe(body: str) -> subprocess.CompletedProcess:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    probe = PROBE_DIR / "test_zsd3_probe.py"
    probe.write_text(textwrap.dedent(body), encoding="utf-8")
    env = dict(os.environ)
    env.pop("LAB_NO_TREE_JANITOR", None)   # exercise the janitor, not the escape hatch
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider"],
        cwd=str(REPO), capture_output=True, text=True, timeout=600, env=env,
    )


def test_uncommitted_work_already_present_survives_a_run():
    """The failure that started this bead: a run must not revert dirty files."""
    original = TARGET.read_text(encoding="utf-8")
    dirty = original + "\n"
    try:
        TARGET.write_text(dirty, encoding="utf-8")     # dirty BEFORE the run starts
        r = _run_probe("def test_noop():\n    assert True\n")
        # the probe must actually have RUN: otherwise the assertions below are
        # satisfied by a pytest that never started (the vacuous-guard shape).
        assert r.returncode == 0, "probe run failed:\n" + (r.stdout + r.stderr)[-1000:]
        assert "1 passed" in (r.stdout + r.stderr), (
            "probe did not run a passing test:\n" + (r.stdout + r.stderr)[-1000:]
        )
        assert TARGET.read_text(encoding="utf-8") == dirty, (
            "the janitor reverted uncommitted work that existed before the run "
            f"({TARGET_REL} was restored from HEAD)"
        )
        assert "TREE POLLUTION" not in (r.stdout + r.stderr), (
            "pre-existing dirt was reported as pollution by this run:\n"
            + (r.stdout + r.stderr)[-1000:]
        )
    finally:
        TARGET.write_text(original, encoding="utf-8")


def test_residue_from_a_run_is_reported_and_left_in_place():
    """A polluting test must FAIL and its residue must SURVIVE, not be repaired."""
    original = TARGET.read_text(encoding="utf-8")
    try:
        r = _run_probe(
            f"""
            from pathlib import Path

            TARGET = Path(r"{TARGET}")


            def test_pollutes():
                TARGET.write_text(TARGET.read_text(encoding="utf-8") + "\\n",
                                  encoding="utf-8")
            """
        )
        out = r.stdout + r.stderr
        assert "TREE POLLUTION" in out, "residue was not reported loudly:\n" + out[-1500:]
        assert "test_pollutes" in out, "the report did not name the polluting node:\n" + out[-1500:]
        assert TARGET.read_text(encoding="utf-8") == original + "\n", (
            "the fixture restored the file -- it must leave residue in place "
            "and report it instead"
        )
        assert r.returncode != 0, (
            "a test that leaves a TRACKED file modified must fail, otherwise the "
            "residue is inherited by later tests and swept into a commit"
        )
    finally:
        TARGET.write_text(original, encoding="utf-8")
