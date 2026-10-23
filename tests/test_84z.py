import json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-84z" / "metrics.json"
def test_case_count():
    m = json.loads(METRICS.read_text())
    assert m["n_cases"] == 12
def test_floor_pass():
    m = json.loads(METRICS.read_text())
    floor=[r for r in m["per_case"] if r["regime"]=="floor"]
    assert all(r["naive_err"]<=0.25 for r in floor)
def test_bw_fail():
    m = json.loads(METRICS.read_text())
    bw=[r for r in m["per_case"] if r["regime"]=="bandwidth" and r["size"]==16777216]
    assert all(r["naive_err"]>0.25 for r in bw)
def test_bytes_pass():
    m = json.loads(METRICS.read_text())
    assert all(r["bytes_err"]<=0.25 for r in m["per_case"])
    assert m["summary"]["bytes_max_err"] <= 0.0676
