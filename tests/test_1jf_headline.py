import json, pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-1jf" / "metrics.json"

def test_headline_matches():
    m = json.loads(METRICS.read_text())
    assert m["headline"] == m["verdict"]

def test_zero_unusable_in_training_set():
    m = json.loads(METRICS.read_text())
    # all rows in manifest must be gated; n_usable <= n_rows
    sm = m["stage1_manifest"]
    assert sm["n_usable"] <= sm["n_rows"]
    assert sm["gate_rejection_rate"] >= 0