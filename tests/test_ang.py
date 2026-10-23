import json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-ang" / "metrics.json"
def test_step0_baseline_matches_pinned():
    m = json.loads(METRICS.read_text())
    assert abs(m["step0_baseline"]["elementwise"]-0.969) < 1e-3
    assert abs(m["step0_baseline"]["matmul"]-0.409) < 1e-3
    assert abs(m["step0_baseline"]["reduce"]-0.473) < 1e-3
def test_levers_present():
    m = json.loads(METRICS.read_text())
    assert "baseline" in m["levers"]
    assert "+l1" in m["levers"]
    assert "+l2" in m["levers"]
    assert "+both" in m["levers"]
def test_verdict_derived():
    m = json.loads(METRICS.read_text())
    assert m["verdict"] == "GAP-REMAINS"
