"""Guard: gate-citations stage is green on main (gpu-sim-dliz)."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gate_citations_green():
    r = subprocess.run(
        [sys.executable, "scripts/check_gate_citations.py"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300,
    )
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out
