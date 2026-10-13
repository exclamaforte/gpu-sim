#!/usr/bin/env python3
"""Tests for 10.4 (CPU): the join reads committed artifacts, asserts the
REPORT-table values (drift guard), and writes the verdict table."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.10.4"))

import join_104  # noqa: E402


class TestJoin(unittest.TestCase):
    def test_table_matches_committed_sources(self):
        out = REPO / "experiments/gpu-sim-u2o.10.4" / "metrics.json"
        self.assertEqual(join_104.main(str(out)), 0)
        t = json.loads(out.read_text())
        ll = t["cells"]["latency->latency"]
        lm = t["cells"]["latency->memory"]
        self.assertEqual((ll["gbt_95"], ll["transformer_91"],
                          ll["roofline_94"]), (0.1072, 0.1438, 0.5638))
        self.assertEqual((lm["gbt_95"], lm["transformer_91"],
                          lm["roofline_94"]), (0.0274, 0.26, 0.6873))
        # ordering verdicts hold numerically, not just as prose
        self.assertTrue(ll["gbt_95"] < ll["transformer_91"] < ll["roofline_94"])
        self.assertTrue(lm["gbt_95"] < lm["transformer_91"] < lm["roofline_94"])
        self.assertEqual(len(t["gaps"]), 4)
        for g in t["gaps"]:
            self.assertTrue(len(g) > 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
