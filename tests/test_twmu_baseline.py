"""Guard: baseline_failures.txt exists and records the 12 known reds (gpu-sim-twmu)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_file_present():
    p = ROOT / "tests" / "baseline_failures.txt"
    assert p.is_file(), "baseline_failures.txt missing"
    lines = p.read_text().splitlines()
    failed = [ln for ln in lines if ln.startswith("FAILED ")]
    assert len(failed) == 12, f"expected 12 baselined failures, got {len(failed)}"
