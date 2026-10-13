#!/usr/bin/env python3
"""Tests for 11.4 (CPU): headline derives from 10.4's table -- ratios
recompute, cells equal, no prose drift possible."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.11.4"))

import headline_114  # noqa: E402

EXP = REPO / "experiments" / "gpu-sim-u2o.11.4"


class TestHeadline(unittest.TestCase):
    def test_derived_not_typed(self):
        self.assertEqual(headline_114.main(str(EXP / "metrics.json")), 0)
        h = json.loads((EXP / "metrics.json").read_text())
        src = json.loads((REPO / "experiments/gpu-sim-u2o.10.4"
                          / "metrics.json").read_text())
        for triple in ("latency->latency", "latency->memory"):
            self.assertEqual(h[triple]["cells"], src["cells"][triple])
            c = h[triple]["cells"]
            self.assertAlmostEqual(
                h[triple]["roofline_over_transformer"],
                round(c["roofline_94"] / c["transformer_91"], 2))
            self.assertAlmostEqual(
                h[triple]["transformer_over_gbt"],
                round(c["transformer_91"] / c["gbt_95"], 2))
        self.assertEqual(h["axes_without_headline"], src["gaps"])
        self.assertIn("NO", h["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
