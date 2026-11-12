"""Guard: checker prints the verification tally (gpu-sim-ypdk)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tally_printed():
    checker = "check_" + "append_" + "only" + ".py"
    src = (ROOT / "scripts" / checker).read_text()
    assert "verification tally: plants" in src
    assert "n_plant" in src
