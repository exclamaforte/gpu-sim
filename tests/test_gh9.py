import json, pathlib
REPO = pathlib.Path(__file__).resolve().parents[1]
METRICS = REPO / "experiments" / "gpu-sim-gh9" / "metrics.json"

def test_verdict_derived():
    m = json.loads(METRICS.read_text())
    lev = m["lever3"]
    passes = sum(1 for v in lev.values() if v <= 0.25)
    expected = "PASS-ALL" if passes == 3 else ("PARTIAL" if passes >= 1 else "GAP-REMAINS")
    assert m["verdict"] == expected

def test_constants_sourced():
    m = json.loads(METRICS.read_text())
    for k, v in m["constants"].items():
        assert "source" in v

def test_overlap_reported():
    m = json.loads(METRICS.read_text())
    assert "shape_overlap_knee_vs_fixture" in m

def test_noise_floor_reported():
    m = json.loads(METRICS.read_text())
    assert "noise_floor" in m
    assert "high_spread_cases" in m["noise_floor"]
