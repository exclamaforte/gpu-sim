#!/usr/bin/env python3
"""Tests for 17.4 (CPU): parity gate recomputes; FAIL pinned with numbers
(a failure is a recordable result -- the test pins the NUMBERS so a future
pass must move them, not the prose)."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.17.4"))

import compare_174  # noqa: E402

EXP = REPO / "experiments" / "gpu-sim-u2o.17.4"


class TestParityGate(unittest.TestCase):
    def test_verdict_with_numbers(self):
        self.assertEqual(compare_174.main(), 0)
        m = json.loads((EXP / "metrics.json").read_text())
        pc = m["per_class"]
        self.assertAlmostEqual(pc["elementwise"]["median_abs_rel_err"],
                               0.969, places=2)
        self.assertAlmostEqual(pc["matmul"]["median_abs_rel_err"],
                               0.409, places=2)
        self.assertAlmostEqual(pc["reduce"]["median_abs_rel_err"],
                               0.473, places=2)
        for cls in ("elementwise", "matmul", "reduce"):
            self.assertFalse(pc[cls]["passes_25pct"])
        self.assertTrue(m["rank_order"]["preserved"])
        self.assertEqual(m["rank_order"]["dominant_measured"], "matmul")
        self.assertEqual(m["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
