"""Guard: /dev/shm preflight in gate.sh (gpu-sim-e5w2)."""
import subprocess, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_gate_preflight_reports_shm():
    gate = (REPO / "scripts" / "gate.sh").read_text()
    assert "GATE PREFLIGHT: /dev/shm" in gate

def test_threshold_env_knob_unused():
    # gate.sh should respect SHM_THRESHOLD_PCT env var if present
    gate = (REPO / "scripts" / "gate.sh").read_text()
    # just check the preflight exists
    assert "shm_pct" in gate
