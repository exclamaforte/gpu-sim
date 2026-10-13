#!/usr/bin/env python3
"""Tests for 11.2 (CPU): predictions artifact schema + failure diagnosis
recomputed from the artifact (not retyped). Inference itself (~2 min CPU)
runs in run.sh; this test verifies the committed outputs."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.metrics_eval import (  # noqa: E402
    calibration_coverage,
    expected_calibration_error,
)

EXP = REPO / "experiments" / "gpu-sim-u2o.11.2"


class TestFailureAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preds = json.loads((EXP / "predictions.json").read_text())
        cls.ana = json.loads((EXP / "failure_analysis.json").read_text())

    def test_schema(self):
        self.assertEqual(len(self.preds), 57)
        need = {"hash", "op", "triple", "regime", "y", "sigma_log",
                "mu", "var", "abs_err"}
        for p in self.preds:
            self.assertTrue(need <= set(p), p)
            self.assertAlmostEqual(p["abs_err"], abs(p["y"] - p["mu"]),
                                   places=9)
            self.assertGreater(p["var"], 0.0)

    def test_worst_concentrated_measurable_memory(self):
        worst = sorted(self.preds, key=lambda p: -p["abs_err"])[:10]
        n = sum(1 for p in worst if p["triple"] == "latency->memory"
                and p["regime"] == "measurable")
        self.assertGreaterEqual(n, 7)
        # exact pin (reviewer recompute): 9/10, rank-10 intruder named.
        # A typed ALL is a typed number -- this fails if prose says ALL.
        self.assertEqual(n, 9)
        self.assertTrue(worst[9]["hash"].startswith("9b474b0e"))
        m = json.loads((EXP / "metrics.json").read_text())
        self.assertEqual(m["worst10_latmem_measurable"], "9/10")
        self.assertTrue(m["worst10_rank10_intruder"]["hash"]
                        .startswith("9b474b0e"))
        # errors anti-correlated with measurement noise: model failure
        self.assertLess(self.ana["err_vs_sigma_corr"], 0.0)

    def test_head_calibration_recomputes(self):
        yt = [p["y"] for p in self.preds]
        mu = [p["mu"] for p in self.preds]
        va = [p["var"] for p in self.preds]
        sg = [p["sigma_log"] for p in self.preds]
        m = json.loads((EXP / "metrics.json").read_text())
        self.assertAlmostEqual(
            calibration_coverage(yt, mu, va, meas_sigma=sg),
            m["coverage_90_head"], places=4)
        self.assertAlmostEqual(
            expected_calibration_error(yt, mu, va, meas_sigma=sg),
            m["ece_1sigma_head"], places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
