"""Guard that experiments/gpu-sim-b1r6/metrics.json fabric band is produced, not typed.

The stated flip ratios / next-rank-1 per cell must equal the values in
docs/model_facts.json -> fabric_precision (which is derived from
error_budget.ranked). A hand-edited disagreement must fail naming both values.
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import check_docs_model_numbers as c


class TestB1r6FabricPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = json.loads((REPO / "docs" / "model_facts.json").read_text())
        cls.metrics = json.loads((REPO / "experiments" / "gpu-sim-b1r6" / "metrics.json").read_text())
        cls.fp = cls.facts["fabric_precision"]

    def test_flip_down_matches(self):
        for cell in self.metrics["cells"]:
            stated = self.metrics["fabric_precision_band"][cell]["flip_down_ratio"]
            computed = self.fp[cell]["term_scale_cp_flip_down_ratio"]
            self.assertEqual(stated, computed,
                            f"{cell} flip_down {stated!r} != computed {computed!r}")

    def test_flip_up_matches(self):
        for cell in self.metrics["cells"]:
            stated = self.metrics["fabric_precision_band"][cell]["flip_up_ratio"]
            computed = self.fp[cell]["term_scale_cp_flip_up_ratio"]
            self.assertEqual(stated, computed,
                            f"{cell} flip_up {stated!r} != computed {computed!r}")

    def test_next_rank1_matches(self):
        for cell in self.metrics["cells"]:
            stated = self.metrics["fabric_precision_band"][cell]["next_rank1_if_pinned"]
            computed = self.fp[cell]["next_rank1_if_pinned"]
            self.assertEqual(stated, computed,
                            f"{cell} next_rank1 {stated!r} != computed {computed!r}")

    def test_measured_sides_match(self):
        for cell in self.metrics["cells"]:
            stated = self.metrics["fabric_precision_band"][cell]["measured_sides"]
            computed = self.fp[cell]["term_scale_cp_measured_sides"]
            self.assertEqual(stated, computed,
                            f"{cell} measured_sides {stated!r} != computed {computed!r}")


if __name__ == "__main__":
    unittest.main()
