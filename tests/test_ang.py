import json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-ang" / "metrics.json"
def test_verdict_present():
    m = json.loads(METRICS.read_text())
    assert m["verdict"] in {"PASS-ALL","PARTIAL","GAP-REMAINS","BOUNDED-RESIDUAL"}
def test_step0_status():
    m = json.loads(METRICS.read_text())
    assert "step0" in m
