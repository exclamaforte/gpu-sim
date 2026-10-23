import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from epic_u2o import op_latency as ol

TABLE = REPO / "experiments" / "gpu-sim-m93" / "mdz_calibration_fixture.json"
SCORED = REPO / "experiments" / "gpu-sim-v4e" / "v4e_scored.json"


def table():
    return ol.load_table(str(TABLE))


def test_in_grid_not_clamped():
    t = table()
    prov = ol.prediction_provenance("relu", size=1024, table=t)
    assert prov["clamped"] is False
    assert prov["clamp_target"] is None

def test_no_size_axis_reports_none():
    # default table has no size axis
    ol._warned.clear()
    prov = ol.prediction_provenance("relu", size=2 ** 20)
    assert prov["clamped"] is None
    assert prov["no_size_axis"] is True


def test_above_grid_clamped_and_warns_once():
    t = table()
    # reset warned set for determinism
    ol._warned.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ol.latency_ms_for("relu", size=2 ** 29, table=t)
        assert len(w) == 1
        assert "out of calibrated range" in str(w[0].message)
        # second call does not warn
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter("always")
            ol.latency_ms_for("relu", size=2 ** 30, table=t)
            assert len(w2) == 0
    prov = ol.prediction_provenance("relu", size=2 ** 29, table=t)
    assert prov["clamped"] is True
    assert prov["clamp_target"] == prov["cal_max"]


def test_above_grid_relu_reproduces_large_error():
    t = table()
    scored = json.loads(SCORED.read_text())
    case = next(c for c in scored["results"]
                if c["op_class"] == "elementwise" and c["size"] == 536870912)
    pred = ol.latency_ms_for("relu", size=case["size"], table=t)
    measured = case["measured_ms"]
    err = abs(pred - measured) / measured
    assert err > 0.5, (err, pred, measured)


def test_gate_numbers_unchanged():
    subprocess.run([sys.executable, str(REPO / "experiments" / "gpu-sim-m93" / "run_gate.py")],
                   check=True, capture_output=True)
    metrics = json.loads((REPO / "experiments" / "gpu-sim-m93" / "metrics.json").read_text())
    assert round(metrics["per_class"]["elementwise"]["median_abs_rel_err"], 3) == 0.025
    assert round(metrics["per_class"]["matmul"]["median_abs_rel_err"], 3) == 0.038
    assert round(metrics["per_class"]["reduce"]["median_abs_rel_err"], 3) == 0.024
