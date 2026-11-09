"""Guard: /dev/shm preflight in gate.sh (gpu-sim-e5w2). Behavioural."""
import subprocess, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_above_threshold_fails():
    env = dict(os.environ)
    env["SHM_THRESHOLD_PCT"] = "0"
    r = subprocess.run(["bash", "scripts/gate.sh", "--fast"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode != 0
    assert "PREFLIGHT FAIL" in (r.stdout + r.stderr)

def test_below_threshold_passes_shm():
    env = dict(os.environ)
    env["SHM_THRESHOLD_PCT"] = "100"
    r = subprocess.run(["bash", "scripts/gate.sh", "--fast"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=300, env=env)
    assert "PREFLIGHT FAIL" not in (r.stdout + r.stderr)


def test_gate_preflight_reports_shm():
    """Ledger alias: preflight line present in gate.sh."""
    gate = (REPO / "scripts" / "gate.sh").read_text()
    assert gate.count("GATE PREFLIGHT: /dev/shm") >= 1
    assert "SHM_THRESHOLD_PCT" in gate
