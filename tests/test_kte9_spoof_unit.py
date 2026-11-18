"""kte9: a spoofed checker-pid marker must not be honoured -- unit arm.

The integration-shaped arm lives in tests/test_append_only.py and goes through
run_script() (a disposable worktree of HEAD). That isolation is right, but it
makes the test unplantable: a mutation_plant applier changes the working tree,
uncommitted, and run_script() checks out HEAD, so the mutated file is not in the
tree the test inspects (gpu-sim-kk5b).

This arm is deliberately subprocess-free: it imports scripts/check_append_only.py
from the tree the suite is running in, so a mutation to _child_mode() IS visible
to it, and the property it asserts is the same one the integration arm asserts --
a marker whose pid is not a live checker ancestor must not switch the checker
into reduced (collect-only) mode.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "cao_unit_kte9", ROOT / "scripts" / "check_append_only.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dead_pid():
    """A pid that existed and is gone: a real number, no live process."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_spoofed_checker_pid_is_not_honoured(monkeypatch):
    cao = _load_checker()
    monkeypatch.setenv(cao.DEPTH_ENV, str(_dead_pid()))
    assert cao._child_mode() is False, (
        "a marker naming a dead pid must not enter collect-only mode")


def test_absent_marker_is_not_child_mode(monkeypatch):
    cao = _load_checker()
    monkeypatch.delenv(cao.DEPTH_ENV, raising=False)
    assert cao._child_mode() is False
