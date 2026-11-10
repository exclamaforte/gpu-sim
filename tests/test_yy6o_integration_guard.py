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
    cao = _load()
    cao.ROOT = str(ROOT)
    assert cao._is_integration_tree() is False
