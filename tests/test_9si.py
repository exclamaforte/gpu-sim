import json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-9si" / "metrics.json"
def test_verdict():
    m = json.loads(METRICS.read_text())
    assert m["verdict"] == "DRIFT-LIMITED"
def test_all_pass():
    m = json.loads(METRICS.read_text())
    for cls, v in m["per_class"].items():
        assert v["median_err_v2"] <= 0.25
def test_cross_check():
    m = json.loads(METRICS.read_text())
    assert m["cross_check_m93"]["match"] is True
def test_case_count():
    m = json.loads(METRICS.read_text())
    assert m["n_scored_cases"] == 25
