import json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-m93" / "metrics.json"
TRACE = REPO / "experiments" / "gpu-sim-m93" / "sim_trace_m93.json"
def test_pinned_medians():
    m = json.loads(METRICS.read_text())
    pc = m["per_class"]
    assert abs(pc["elementwise"]["median_abs_rel_err"]-0.025) < 1e-3
    assert abs(pc["matmul"]["median_abs_rel_err"]-0.038) < 1e-3
    assert abs(pc["reduce"]["median_abs_rel_err"]-0.024) < 1e-3
def test_pinned_max():
    sim = json.loads(TRACE.read_text())
    for cls, exp in [("elementwise",0.104),("matmul",0.175),("reduce",0.196)]:
        errs=[abs(d["fit_rel_err"]) for d in sim["classes"][cls]["held_out"]]
        assert max(errs) < exp+1e-3
def test_verdict_pass():
    m = json.loads(METRICS.read_text())
    assert m["verdict"] == "PASS"
def test_trace_artifact():
    assert TRACE.exists()
def test_disjointness_stated():
    m = json.loads(METRICS.read_text())
    assert "calibration_sha256" in m
