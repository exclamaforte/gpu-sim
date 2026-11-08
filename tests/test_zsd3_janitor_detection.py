"""gpu-sim-zsd3: the tree janitor must DETECT residue, never repair it.

Why a subprocess: the thing under test is the autouse fixture's TEARDOWN, and no
test can honestly observe its own teardown (it has not happened yet). Each case
therefore runs a real pytest on a tiny probe file and inspects the effect on the
repository afterwards.

Why a SCRATCH WORKTREE: the probe has to leave a TRACKED file dirty, and doing
that in the live tree is the defect this bead is about -- two concurrent runs in
one tree both fail and leave ' M docs/model_facts.json' behind (demonstrated by
the reviewer in a clone). Every case gets its own disposable worktree under
.lab/tmp, so dirt cannot escape and concurrent runs cannot collide.

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
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TARGET_REL = "docs/model_facts.json"


def _run(cmd, cwd):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=600)


@pytest.fixture()
def sandbox():
    """A detached worktree carrying the WORKING-TREE conftest (the code under test).

    The checkout comes from HEAD, then conftest.py is overlaid from the live
    worktree: the fixture code under review is whatever is on disk now, while
    everything it dirties stays inside the sandbox.
    """
    d = tempfile.mkdtemp(prefix="zsd3-probe-", dir=os.path.join(REPO, ".lab", "tmp"))
    add = _run(["git", "worktree", "add", "--detach", d, "HEAD"], REPO)
    if add.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        pytest.skip(f"cannot create scratch worktree: {add.stderr}")
    try:
        shutil.copy(REPO / "conftest.py", Path(d) / "conftest.py")
        # conftest's pytest_configure runs the libfakegpu.so freshness guard,
        # and the .so is an UNTRACKED build artifact so a fresh worktree does not
        # have one. Copy it and touch it: the sandbox exists to exercise the
        # janitor, not to re-prove build freshness, and a checked-out source is
        # newer than any real build, so an untouched copy would read as stale.
        so = REPO / "libfakegpu.so"
        if so.exists():
            shutil.copy2(so, Path(d) / "libfakegpu.so")
            os.utime(Path(d) / "libfakegpu.so", None)
        yield Path(d)
    finally:
        _run(["git", "worktree", "remove", "--force", d], REPO)
        _run(["git", "worktree", "prune"], REPO)
        shutil.rmtree(d, ignore_errors=True)


def _run_probe(sandbox: Path, body: str) -> subprocess.CompletedProcess:
    probe_dir = sandbox / "probe"
    probe_dir.mkdir(exist_ok=True)
    probe = probe_dir / "test_zsd3_probe.py"
    probe.write_text(textwrap.dedent(body), encoding="utf-8")
    env = dict(os.environ)
    env.pop("LAB_NO_TREE_JANITOR", None)   # exercise the janitor, not the escape hatch
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider"],
        cwd=str(sandbox), capture_output=True, text=True, timeout=600, env=env,
    )


def test_uncommitted_work_already_present_survives_a_run(sandbox):
    """The failure that started this bead: a run must not revert dirty files."""
    target = sandbox / TARGET_REL
    original = target.read_text(encoding="utf-8")
    dirty = original + "\n"
    target.write_text(dirty, encoding="utf-8")     # dirty BEFORE the run starts

    r = _run_probe(sandbox, "def test_noop():\n    assert True\n")
    out = r.stdout + r.stderr
    # the probe must actually have RUN: otherwise the assertions below are
    # satisfied by a pytest that never started (the vacuous-guard shape).
    # '1 passed in' rather than '1 passed': that substring also matches '11 passed'.
    assert r.returncode == 0, "probe run failed:\n" + out[-1000:]
    assert re.search(r"\b1 passed in ", out), "probe did not run exactly one passing test:\n" + out[-1000:]
    assert target.read_text(encoding="utf-8") == dirty, (
        "the janitor reverted uncommitted work that existed before the run "
        f"({TARGET_REL} was restored from HEAD)"
    )
    assert "TREE POLLUTION" not in out, (
        "pre-existing dirt was reported as pollution by this run:\n" + out[-1000:]
    )


def test_residue_from_a_run_is_reported_and_left_in_place(sandbox):
    """A polluting test must FAIL and its residue must SURVIVE, not be repaired."""
    target = sandbox / TARGET_REL
    original = target.read_text(encoding="utf-8")

    r = _run_probe(
        sandbox,
        f"""
        from pathlib import Path

        TARGET = Path(r"{target}")


        def test_pollutes():
            TARGET.write_text(TARGET.read_text(encoding="utf-8") + "\\n",
                              encoding="utf-8")
        """,
    )
    out = r.stdout + r.stderr
    assert "TREE POLLUTION" in out, "residue was not reported loudly:\n" + out[-1500:]
    assert "test_pollutes" in out, "the report did not name the polluting node:\n" + out[-1500:]
    assert target.read_text(encoding="utf-8") == original + "\n", (
        "the fixture restored the file -- it must leave residue in place and "
        "report it instead"
    )
    assert r.returncode != 0, (
        "a test that leaves a TRACKED file modified must fail, otherwise the "
        "residue is inherited by later tests and swept into a commit"
    )
