"""Guard: gate.sh manifest derives EXPECTED_HARD from run_hard count (gpu-sim-8nv5)."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "scripts" / "gate.sh"


def test_expected_hard_is_derived():
    text = GATE.read_text()
    # EXPECTED_HARD must be derived from HARD_COUNT, not a literal integer
    assert "EXPECTED_HARD=$HARD_COUNT" in text, (
        "gate.sh must derive EXPECTED_HARD from HARD_COUNT, not a literal")
    import re
    assert not re.search(r"EXPECTED_HARD=\d+", text), (
        "gate.sh must not hardcode EXPECTED_HARD to a literal integer")


def test_hard_count_incremented_in_run_hard():
    text = GATE.read_text()
    assert "HARD_COUNT=$((HARD_COUNT + 1))" in text, (
        "run_hard must increment HARD_COUNT")
