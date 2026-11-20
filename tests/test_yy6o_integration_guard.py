"""yy6o: the walk refuses to run in the integration tree.

Running check_append_only in the lab root mutates TRACKED files there, so any
concurrent gate that snapshots the root sees "uncommitted changes (different
work)" and fails. That happened three times tonight (trcz, ccw5, 20ms) before
this guard existed.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / ("check_append" + "_only.py")
INTEGRATION = Path("/home/gabe/Documents/gpu-sim")


def _load():
    spec = importlib.util.spec_from_file_location("cao_yy6o", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_integration_tree_is_detected():
    cao = _load()
    cao.ROOT = str(INTEGRATION)
    assert cao._is_integration_tree() is True


def test_bead_worktree_is_not_the_integration_tree():
    """A REAL linked worktree must not be mistaken for the integration tree.

    gpu-sim-94pp: this used to set cao.ROOT = str(ROOT) and assert False. That holds only when
    the suite itself runs inside a bead worktree; the gate runs it in MAIN, where ROOT IS the
    integration tree, so the assertion was false by construction and main's full gate went red
    while every reviewer worktree saw green. The test measured where it was started, not what
    the detector does. It now builds its own world: a disposable linked worktree under
    .lab/tmp (never /tmp -- gpu-sim-u6j5) and asks the detector about THAT.
    """
    import shutil
    import subprocess
    import tempfile

    cao = _load()
    scratch = Path(tempfile.mkdtemp(prefix="yy6o-wt-", dir=str(ROOT / ".lab" / "tmp")))
    wt = scratch / "wt"
    try:
        add = subprocess.run(
            ["git", "-C", str(INTEGRATION), "worktree", "add", "--detach", str(wt), "HEAD"],
            capture_output=True, text=True, timeout=120)
        assert add.returncode == 0, f"worktree add failed: {add.stderr}"
        cao.ROOT = str(wt)
        assert cao._is_integration_tree() is False, (
            f"a linked worktree at {wt} was reported as the integration tree")
    finally:
        subprocess.run(["git", "-C", str(INTEGRATION), "worktree", "remove", "--force", str(wt)],
                       capture_output=True, text=True, timeout=120)
        subprocess.run(["git", "-C", str(INTEGRATION), "worktree", "prune"],
                       capture_output=True, text=True, timeout=120)
        shutil.rmtree(scratch, ignore_errors=True)
