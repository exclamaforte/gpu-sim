"""Guard gpu-sim-006j: act_factor shape dependence anchored as produced facts."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import error_budget


class Test006jActShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = json.loads((REPO / "experiments" / "gpu-sim-006j" / "analysis.json").read_text())
        cls.facts = json.loads((REPO / "docs" / "model_facts.json").read_text())
        cls.mas = cls.facts["scaleout_opt"]["measured_act_shape"]

    def test_curves_match_analysis(self):
        self.assertEqual(self.mas["curve_act_factor_by_seq"],
                         self.analysis["curve_act_factor_by_seq"])

    def test_shape_dependence_matches_analysis(self):
        self.assertEqual(self.mas["shape_dependence"],
                         self.analysis["shape_dependence"])

    def test_converges_at_long_seq(self):
        sd = self.mas["shape_dependence"]
        self.assertLess(sd["seq16384"]["spread_pct"], 2.0)
        self.assertLess(sd["seq32768"]["spread_pct"], 2.0)

    def test_diverges_at_short_seq(self):
        sd = self.mas["shape_dependence"]
        self.assertGreater(sd["seq2048"]["spread_pct"], 40.0)

    def test_three_shapes_present(self):
        curves = self.mas["curve_act_factor_by_seq"]
        self.assertEqual(len(curves), 3)
        for shape, seqs in curves.items():
            self.assertEqual(len(seqs), 5)

    def test_deciding_cell_value(self):
        # Value check only: the deciding cell (N16_seq32768) measured value is
        # 2.3448 (orig shape at 32768). The contract that error_budget actually
        # uses this value (and not a contrasting shape's) is guarded separately
        # by test_error_budget_curve_tied_to_facts.
        orig_32768 = self.mas["curve_act_factor_by_seq"]["8L_d1024_dff4096"]["32768"]
        self.assertAlmostEqual(orig_32768, 2.3448, places=3)

    def test_error_budget_curve_tied_to_facts(self):
        # Fast guard: scripts/error_budget.py::_ACT_FACTOR_BY_SEQ must equal the
        # facts' original-shape curve (int keys in code vs str keys in facts).
        # Planting a contrasting shape's value (e.g. 2.3373 at 32768) must fail.
        orig = self.mas["curve_act_factor_by_seq"]["8L_d1024_dff4096"]
        expected = {int(k): v for k, v in orig.items()}
        self.assertEqual(error_budget._ACT_FACTOR_BY_SEQ, expected)


if __name__ == "__main__":
    unittest.main()
