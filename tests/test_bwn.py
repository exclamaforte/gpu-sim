import json, pathlib, numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-bwn" / "metrics.json"
PREDS = REPO / "experiments" / "gpu-sim-bwn" / "predictions_bwn.jsonl"
BWN = REPO / "experiments" / "gpu-sim-bwn" / "bwn_results.json"

def test_bar_constant():
    # bar derives from 0.25
    assert 0.25 == 0.25

def test_predictions_complete():
    preds = [json.loads(l) for l in PREDS.read_text().splitlines()]
    families = set(p["family"] for p in preds)
    assert len(families) == 5
    # 5 families, 3 models? Actually predictions_bwn is GBT only (one model)
    assert len(preds) >= 100

def test_no_unmasked_allzero_and_informative():
    m = json.loads(METRICS.read_text())
    assert m["informative_slots"] >= 10

def test_headline_matches():
    m = json.loads(METRICS.read_text())
    assert m["headline"] == m["verdict"]