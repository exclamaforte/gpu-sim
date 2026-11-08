"""Guard: /dev/shm preflight in gate.sh (gpu-sim-e5w2). Behavioural."""
import subprocess, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_preflight_line_present():
    gate = (REPO / "scripts" / "gate.sh").read_text()
    assert "GATE PREFLIGHT: /dev/shm" in gate

def test_threshold_env_knob():
    gate = (REPO / "scripts" / "gate.sh").read_text()
    assert "SHM_THRESHOLD_PCT" in gate

def test_above_threshold_fails():
    # Set threshold to 0 so any usage triggers failure
    env = dict(os.environ)
    env["SHM_THRESHOLD_PCT"] = "0"
    r = subprocess.run(["bash", "scripts/gate.sh", "--fast"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=300, env=env)
    # /dev/shm is never 0% used, so this should fail
    assert r.returncode != 0, f"expected failure with threshold 0: {r.stdout[:300]}"
    assert "PREFLIGHT FAIL" in (r.stdout + r.stderr), f"expected PREFLIGHT FAIL: {r.stdout[:300]}"

def test_below_threshold_passes():
    env = dict(os.environ)
    env["SHM_THRESHOLD_PCT"] = "100"
    r = subprocess.run(["bash", "scripts/gate.sh", "--fast"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=300, env=env)
    # With threshold 100, should not fail on shm
    assert "PREFLIGHT FAIL" not in (r.stdout + r.stderr), f"unexpected PREFLIGHT FAIL: {r.stdout[:300]}"
