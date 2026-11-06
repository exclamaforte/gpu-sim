"""Guard: conftest tree janitor is present (gpu-sim-zsd3)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_janitor_present():
    src = (REPO / "conftest.py").read_text()
    assert "tree janitor" in src
