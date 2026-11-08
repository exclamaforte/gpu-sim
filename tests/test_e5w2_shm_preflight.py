"""Guard: gate.sh preflight reports /dev/shm utilisation (gpu-sim-e5w2)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_gate_preflight_reports_shm():
    gate = (REPO / "scripts" / "gate.sh").read_text()
    assert "GATE PREFLIGHT: /dev/shm" in gate
    assert "shm_pct" in gate
