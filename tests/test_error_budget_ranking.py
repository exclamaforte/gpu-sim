"""Guard the error-budget ranking (gpu-sim-hd5u).

The ranking column (fraction_one_sided) is the decision input for the wv2
measurement program. check_docs_model_numbers.py only re-runs the script and
embeds its output, so a wrong sort key propagates into facts and doc together
and that check still passes. This test pins the anchors and the top row so a
regression in the sort key fails presubmit.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import error_budget


def _row(ranked, param, verdict):
    for r in ranked:
        if r["parameter"] == param and r["affected_verdict"] == verdict:
            return r
    raise AssertionError(f"no row for {param} / {verdict}")


class TestErrorBudgetRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ranked, cls.unranked, cls.result = error_budget.compute()

    def test_top_row(self):
        self.assertEqual(self.ranked[0]["parameter"], "microbatch_factor")
        self.assertEqual(self.ranked[0]["affected_verdict"], "N8_winner")

    def test_cp_n8_both_crossings(self):
        r = _row(self.ranked, "term_scale_cp", "N8_winner")
        self.assertAlmostEqual(r["flip_down"], 0.541452, places=5)
        self.assertAlmostEqual(r["flip_up"], 1.524446, places=5)

    def test_pp_n8_anchor(self):
        r = _row(self.ranked, "term_scale_pp", "N8_winner")
        self.assertAlmostEqual(r["flip_value"], 0.825536, places=5)

    def test_microbatch_anchor(self):
        r = _row(self.ranked, "microbatch_factor", "N8_winner")
        self.assertAlmostEqual(r["flip_value"], 1.416498, places=5)

    def test_cp_n16_anchor(self):
        r = _row(self.ranked, "term_scale_cp", "N16_winner")
        self.assertAlmostEqual(r["flip_value"], 0.538343, places=5)

    def test_pp_n16_anchor(self):
        r = _row(self.ranked, "term_scale_pp", "N16_winner")
        self.assertAlmostEqual(r["flip_value"], 1.857551, places=5)

    def test_bw_anchor(self):
        r = _row(self.ranked, "achieved_bw_scale", "N8_winner")
        self.assertAlmostEqual(r["flip_value"], 0.814132, places=5)

    def test_nulls_sorted_last(self):
        # all rows with no flip must come after every ranked flip row
        seen_null = False
        for r in self.ranked:
            if r["flip_value"] is None:
                seen_null = True
            else:
                self.assertFalse(seen_null, "a flip row appeared after a null row")

    def test_measured_sides_contract(self):
        for r in self.ranked:
            ms = r["measured_sides"]
            self.assertIsInstance(ms, list)
            self.assertEqual(ms, sorted(ms))
            for s in ms:
                self.assertIn(s, ("up", "down"))


if __name__ == "__main__":
    unittest.main()
