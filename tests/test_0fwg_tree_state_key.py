"""Guard: tree_state uses reachable_from_trunk key (gpu-sim-0fwg)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_key_renamed():
    src = (ROOT / "scripts" / ("tree_state" + ".py")).read_text()
    assert "reachable_from_trunk" in src
