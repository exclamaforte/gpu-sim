"""2fo4: --fast must name the stage it skips."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "gate.sh"


def test_fast_mode_declares_skipped_suite():
    text = GATE.read_text()
    assert "SKIPPED: suite (pytest) -- fast mode" in text


def test_fast_summary_names_skipped():
    text = GATE.read_text()
    assert "1 skipped (suite)" in text
