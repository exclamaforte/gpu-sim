import json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-7l9" / "metrics.json"
def test_unseen_ops():
    m = json.loads(METRICS.read_text())
    assert m["unseen_ops"] == ["gelu","hardsigmoid"]
    assert m["verified_absent_from_table"] is True
def test_all_pass():
    m = json.loads(METRICS.read_text())
    for r in m["per_case"]:
        assert r["relu_err"] <= 0.25
        assert r["class_err"] <= 0.25
def test_case_count():
    m = json.loads(METRICS.read_text())
    assert m["n_cases"] == 8
