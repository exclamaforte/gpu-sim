import json, pathlib, numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-2al" / "metrics.json"
DATASET = REPO / "experiments" / "gpu-sim-2al" / "dataset.jsonl"

def test_no_unmasked_all_zero_slot():
    m = json.loads(METRICS.read_text())
    wc = m["wiring_check"]
    masked = set(m["dataset"]["features_masked"])
    # any slot all-zero in training set must be in masked
    rows = [json.loads(l) for l in DATASET.read_text().splitlines()]
    X = np.array([r["num33"] for r in rows])
    allzero = {i for i in range(33) if (X[:,i]==0).all()}
    assert allzero.issubset(masked), (allzero - masked)

def test_zero_unusable_rows():
    rows = [json.loads(l) for l in DATASET.read_text().splitlines()]
    assert all(r.get("target_usable") for r in rows)

def test_headline_matches():
    m = json.loads(METRICS.read_text())
    assert m["headline"] == m["verdict"]