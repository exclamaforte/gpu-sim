#!/usr/bin/env python3
"""Tests for 17.3 (CPU): scaling-law fit recomputes from the 17.5 fixture;
naive comparison and gaps pinned."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.17.3"))

import sim_trace_173 as S  # noqa: E402

EXP = REPO / "experiments" / "gpu-sim-u2o.17.3"


class TestSimTrace(unittest.TestCase):
    def test_fit_beats_naive_where_smooth(self):
        self.assertEqual(S.main(str(EXP / "metrics.json")), 0)
        m = json.loads((EXP / "metrics.json").read_text())
        mat = m["classes"]["matmul"]
        red = m["classes"]["reduce"]
        ele = m["classes"]["elementwise"]
        self.assertLess(mat["mean_abs_err_fit"], mat["mean_abs_err_naive"])
        self.assertLess(red["mean_abs_err_fit"], red["mean_abs_err_naive"])
        # elementwise knee: single-slope loses (pinned -- two-regime is
        # the follow-up, not a silent tweak of these numbers)
        self.assertGreater(ele["mean_abs_err_fit"], ele["mean_abs_err_naive"])
        self.assertAlmostEqual(mat["mean_abs_err_fit"], 0.333, places=2)
        self.assertAlmostEqual(red["mean_abs_err_fit"], 0.460, places=2)

    def test_singleton_gaps_recorded(self):
        m = json.loads((EXP / "metrics.json").read_text())
        self.assertEqual(len(m["gaps"]), 2)
        self.assertTrue(any("chain" in g for g in m["gaps"]))
        self.assertTrue(any("launch_bound" in g for g in m["gaps"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
