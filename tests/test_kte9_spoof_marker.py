"""Unit test for spoofed OI4M_CHECKER_PID rejection (gpu-sim-kte9)."""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / ("check_append" + "_only.py")


def _load():
    spec = importlib.util.spec_from_file_location("cao_kte9_unit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_spoofed_depth_marker_rejected():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = proc.pid
    proc.wait(timeout=10)
    cao = _load()
    old = os.environ.get(cao.DEPTH_ENV)
    try:
        os.environ[cao.DEPTH_ENV] = str(dead_pid)
        assert cao._child_mode() is False
    finally:
        if old is None:
            os.environ.pop(cao.DEPTH_ENV, None)
        else:
            os.environ[cao.DEPTH_ENV] = old
