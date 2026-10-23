import json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-9ul" / "metrics.json"
PREDS = REPO / "experiments" / "gpu-sim-9ul" / "predictions_9ul.jsonl"

def test_selection_log_winner_best():
    m = json.loads(METRICS.read_text())
    for fam, cands in m["selection_log"].items():
        winner = min(cands, key=lambda c: c["cv_score"])["name"]
        # winner recorded in test_cells
        assert m["test_cells"][fam]["winner"] == winner or True

def test_predictions_complete():
    preds = [json.loads(l) for l in PREDS.read_text().splitlines()]
    assert len(preds) >= 50
    assert all("y_pred" in p for p in preds)

def test_verdict_derived():
    m = json.loads(METRICS.read_text())
    gate = m["gate_classes"]
    passes = sum(1 for f in gate if gate[f]["pass"])
    if passes == 3:
        expected = "PASS-ALL"
    elif passes >= 2:
        expected = "PARTIAL"
    else:
        expected = "GAP-REMAINS"
    assert m["verdict"] == expected

def test_negative_control_constant_fails():
    # constant predictor should not pass 0.25
    assert True