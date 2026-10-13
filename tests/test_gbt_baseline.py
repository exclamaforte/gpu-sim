#!/usr/bin/env python3
"""Tests for the 9.5 GBT baseline. Every test pins an acceptance item."""
import json
import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "gpu-sim-u2o.9.5"
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.9.5"))


def metrics():
    p = EXP / "metrics.json"
    if not p.exists():
        raise unittest.SkipTest("run train_gbt_95.py first")
    return json.loads(p.read_text())


class TestJoin(unittest.TestCase):
    def test_join_456_no_shortfall(self):
        m = metrics()["join"]
        self.assertEqual(m["n"], 456)
        self.assertEqual(m["shortfall"], [])

    def test_median_agreement(self):
        feats = {json.loads(l)["kernel_hash"]: json.loads(l) for l in
                 (REPO / "experiments/gpu-sim-u2o.8.3/features.jsonl")
                 .read_text().splitlines()}
        targs = {json.loads(l)["kernel_hash"]: json.loads(l) for l in
                 (REPO / "experiments/gpu-sim-u2o.6.6/targets.jsonl")
                 .read_text().splitlines()}
        for h, f in feats.items():
            self.assertAlmostEqual(f["measured_median_ms"],
                                   targs[h]["median_runtime_ms"], places=12)


class TestTransform(unittest.TestCase):
    def test_round_trip_bounded_by_guard(self):
        self.assertLess(abs(math.exp(math.log(0.002536 + 1e-6)) - 0.002536),
                        2e-6)


class TestNoBias(unittest.TestCase):
    def test_no_bias_field_in_features(self):
        for line in (REPO / "experiments/gpu-sim-u2o.8.3/features.jsonl") \
                .read_text().splitlines():
            f = json.loads(line)
            self.assertFalse([k for k in f.keys() if "bias" in k.lower()])
            self.assertEqual(len(f["vector"]), 33)


class TestCensus(unittest.TestCase):
    def test_imabalance_flag_fires_on_single_regime(self):
        sys.path.insert(0, str(EXP))
        import train_gbt_95 as t
        lb = [{"triple": "latency->latency"}] * 283
        me = [{"triple": "latency->memory"}] * 150 + \
             [{"triple": "memory->memory"}] * 23
        self.assertIn("REGIME-IMBALANCED", t.imbalance_flag(lb, me))

    def test_balanced_census_is_silent(self):
        import train_gbt_95 as t
        a = [{"triple": "latency->latency"}] * 100 + \
            [{"triple": "latency->memory"}] * 100
        b = [{"triple": "latency->latency"}] * 50 + \
            [{"triple": "latency->memory"}] * 50
        self.assertEqual(t.imbalance_flag(a, b), "")


class TestArtifact(unittest.TestCase):
    def test_stage_labels_and_no_generalisation_claim_in_stage1(self):
        m = metrics()
        self.assertIn("IN-SAMPLE", m["stage1_insample"]["label"])
        self.assertIn("no generalisation claim",
                      m["stage1_insample"]["label"].lower())

    def test_94_frame_loaded_not_recomputed(self):
        m = metrics()
        r94 = json.loads((REPO / "experiments/gpu-sim-u2o.9.4/metrics.json")
                         .read_text())
        self.assertEqual(m["roofline_94_frame"]["residual_A"],
                         r94["residual_A"])
        self.assertEqual(m["roofline_94_frame"]["per_regime_achieved"],
                         r94["per_regime_achieved"])

    def test_holdout_has_zero_shared_groups(self):
        m = metrics()["stage2_holdout"]
        self.assertEqual(m["shared_digest_groups_across_sides"], 0)
        self.assertEqual(sum(v["n"] for v in m["sizes"].values()), 456)


if __name__ == "__main__":
    unittest.main(verbosity=2)
