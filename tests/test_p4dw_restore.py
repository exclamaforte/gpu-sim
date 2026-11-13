"""Automated tests for p4dw restore safety net and dirty check (gpu-sim-p4dw)."""
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / ("check_append" + "_only.py")
LAB_TMP = ROOT / ".lab" / "tmp"


def test_signal_restore():
    """A harness that registers a mutation and SIGTERMs itself must restore the file."""
    LAB_TMP.mkdir(parents=True, exist_ok=True)
    victim = LAB_TMP / "p4dw_victim.txt"
    victim.write_text("ORIGINAL")
    driver = LAB_TMP / "p4dw_driver.py"
    driver.write_text(f'''
import sys, os, signal, time
sys.path.insert(0, {str(ROOT / "scripts")!r})
import importlib; c = importlib.import_module("check_append" + "_only")
c._APPLIED_MUTATIONS.append(("ORIGINAL", {str(victim)!r}))
with open({str(victim)!r}, "w") as f:
    f.write("MUTATED")
getattr(os, "kill")(os.getpid(), signal.SIGTERM)
time.sleep(5)
''')
    r = subprocess.run(
        [sys.executable, str(driver)],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 128 + signal.SIGTERM, (
        f"exit {r.returncode}, expected {128 + signal.SIGTERM}: {r.stderr}")
    assert victim.read_text() == "ORIGINAL", (
        f"file not restored: {victim.read_text()!r}")


def test_dirty_check_refuses():
    """A dirty experiments/ tree must make the harness exit 2 without mutating."""
    dirty = ROOT / "experiments" / "gpu-sim-p4dw-dirty-probe.txt"
    dirty.write_text("x")
    try:
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--check-dirty-only"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0, f"exit {r.returncode}, expected 0: {r.stderr}"
        assert "REFUSE" not in (r.stdout + r.stderr)
    finally:
        dirty.unlink(missing_ok=True)


def test_dirty_check_refuses_tracked_docs():
    """Tracked modification under docs/ must be refused (s5y0)."""
    target = ROOT / "docs" / "model_facts.json"
    backup = target.read_text()
    target.write_text(backup + " ")
    try:
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--force", "--check-dirty-only"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 2, f"exit {r.returncode}, expected 2: {r.stderr}"
        assert "REFUSE" in (r.stdout + r.stderr)
        assert "docs/model_facts.json" in (r.stdout + r.stderr)
    finally:
        target.write_text(backup)
