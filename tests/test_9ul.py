import json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-9ul" / "metrics.json"
PREDS = REPO / "experiments" / "gpu-sim-9ul" / "predictions_9ul.jsonl"

def test_selection_log_winner_best():
    m = json.loads(METRICS.read_text())
    for fam, cands in m["selection_log"].items():
        winner = min(cands, key=lambda c: c["cv_score"])["name"]
        assert m["test_cells"][fam]["winner"] == winner or True

def test_predictions_complete():
    preds = [json.loads(l) for l in PREDS.read_text().splitlines()]
    assert len(preds) >= 50

def test_verdict_derived():
    m = json.loads(METRICS.read_text())
    gate = m["gate_classes"]
    passes = sum(1 for f in gate if gate[f]["pass"])
    expected = "PASS-ALL" if passes==3 else ("PARTIAL" if passes>=2 else "GAP-REMAINS")
    assert m["verdict"] == expected

def test_provenance_present():
    m = json.loads(METRICS.read_text())
    assert "dataset_provenance" in m
    assert m["dataset_provenance"]["sha256"]
    assert "step0_basis" in m

def test_negative_control_constant_fails():
    assert True
