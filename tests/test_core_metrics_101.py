#!/usr/bin/env python3
"""Tests for 10.1 core metrics (CPU). Pins the variance-contract correction
and runs every metric on REAL rows (constant train-mean baseline on the
family_holdout test split -- the audit's 'never run on real data' point).
"""
import json
import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))

from epic_u2o.metrics_eval import (  # noqa: E402
    calibration_coverage,
    core_metrics,
    expected_calibration_error,
    mape,
    r2,
    rmse,
    rmsle,
)
import train_trans_91 as T  # noqa: E402

EXP = REPO / "experiments" / "gpu-sim-u2o.10.1"


class TestUnits(unittest.TestCase):
    def test_perfect(self):
        y = [1.0, 2.0, 3.0]
        self.assertEqual(mape(y, y), 0.0)
        self.assertEqual(rmsle(y, y), 0.0)
        self.assertEqual(r2(y, y), 1.0)

    def test_hand_values(self):
        self.assertAlmostEqual(mape([100.0], [110.0]), 10.0)
        self.assertAlmostEqual(rmse([0.0, 0.0], [3.0, 4.0]), math.sqrt(12.5))

    def test_variance_not_logvar(self):
        # y=1.0, mean=0, var=0.25: std=0.5 -> 1.0 > 1.645*0.5 MISS (0.0).
        # Under the old exp() reading std would be 1.136 -> HIT (1.0).
        c = calibration_coverage([1.0], [0.0], [0.25])
        self.assertEqual(c, 0.0)

    def test_meas_sigma_widens_interval(self):
        c0 = calibration_coverage([1.0], [0.0], [1e-9])
        c1 = calibration_coverage([1.0], [0.0], [1e-9], meas_sigma=1.0)
        self.assertEqual(c0, 0.0)
        self.assertEqual(c1, 1.0)

    def test_ece_range(self):
        e = expected_calibration_error([0.0] * 20, [0.0] * 20, [1.0] * 20)
        self.assertTrue(0.0 <= e <= 1.0)


class TestRealData(unittest.TestCase):
    def test_constant_baseline_on_test_split(self):
        rows = [json.loads(l) for l in
                (REPO / "experiments/gpu-sim-u2o.9.1"
                 / "train_data_91.jsonl").read_text().splitlines()]
        parts, _, _ = T.family_holdout_split(rows, REPO)
        self.assertEqual([len(parts[k]) for k in ("train", "val", "test")],
                         [342, 57, 57])
        import statistics
        tr_y = [r["y"] for r in parts["train"]]
        mu, va = statistics.mean(tr_y), statistics.variance(tr_y)
        te = parts["test"]
        yt = [r["y"] for r in te]
        yp = [mu] * len(te)
        sig = [r["sigma_log"] for r in te]
        log_m = core_metrics(yt, yp, mean=yp, var=[va] * len(te),
                             meas_sigma=sig)
        raw_t = [math.exp(v) for v in yt]
        raw_p = [math.exp(mu)] * len(te)
        raw_m = core_metrics(raw_t, raw_p)
        self.assertTrue(0.0 <= log_m["coverage_90"] <= 1.0)
        self.assertTrue(0.0 <= log_m["ece_1sigma"] <= 1.0)
        metrics = {"bead": "gpu-sim-u2o.10.1",
                   "baseline": "train-mean constant (log), train-var cal.",
                   "n_test": len(te),
                   "log_space": {k: round(v, 4) if isinstance(v, float)
                                 else v for k, v in log_m.items()},
                   "raw_ms": {k: round(v, 4) if isinstance(v, float)
                              else v for k, v in raw_m.items()}}
        (EXP / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("\n10.1 metrics:", json.dumps(metrics))


if __name__ == "__main__":
    unittest.main(verbosity=2)
