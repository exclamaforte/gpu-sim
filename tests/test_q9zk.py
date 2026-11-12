"""Regression: q9zk re-run flags the negative kernel_us row (gpu-sim-q9zk)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "gpu-sim-q9zk" / "result.json"


def test_q9zk_strict_flags_negative_row():
    data = json.loads(RESULT.read_text())
    assert data["exit_code"] == 1, "strict run must exit non-zero"
    assert data["non_positive_count"] >= 1, "negative row must be counted"
    nr = data["negative_row"]
    assert nr is not None
    assert nr["kernel_us"] <= 0
    assert nr["target_usable"] is False
    assert nr["rejection_reason"] == "non_positive_kernel"
    assert nr["log_kernel_us"] is None
    assert "rows with non-positive kernel_us" in data["done_line"]
