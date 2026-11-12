"""Regression: q9zk re-run flags the negative kernel_us row (gpu-sim-q9zk)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "experiments" / "gpu-sim-q9zk" / "measured_rows_sizescale_strict.jsonl"
RESULT = ROOT / "experiments" / "gpu-sim-q9zk" / "result.json"
METRICS = ROOT / "experiments" / "gpu-sim-q9zk" / "metrics.json"


def test_q9zk_strict_flags_negative_row():
    rows = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    neg = [r for r in rows if r.get("rejection_reason") == "non_positive_kernel"]
    # exactly one non-positive row
    assert len(neg) == 1, f"expected 1 non-positive row, got {len(neg)}"
    nr = neg[0]
    assert nr["kernel_us"] <= 0
    assert nr["target_usable"] is False
    assert nr["rejection_reason"] == "non_positive_kernel"
    assert nr["log_kernel_us"] is None
    # cross-check result.json / metrics.json against derived numbers
    res = json.loads(RESULT.read_text())
    assert res["total_rows"] == len(rows)
    assert res["non_positive_count"] == len(neg)
    assert res["negative_row"]["kernel_us"] == nr["kernel_us"]
    assert res["negative_row"]["target_usable"] == nr["target_usable"]
    assert res["negative_row"]["rejection_reason"] == nr["rejection_reason"]
    assert res["negative_row"]["log_kernel_us"] == nr["log_kernel_us"]
    assert res["exit_code"] == 1
    met = json.loads(METRICS.read_text())
    assert met["total_rows"] == len(rows)
    assert met["non_positive_count"] == len(neg)
    assert met["negative_kernel_us"] == nr["kernel_us"]
    assert met["strict_exit_code"] == 1
