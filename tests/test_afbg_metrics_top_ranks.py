"""Guard that experiments/gpu-sim-afbg/metrics.json 'top_ranks' is produced, not typed.

The stated rank-1 per cell must equal the computed ranked[0] (top_param_for_cell)
derived from docs/model_facts.json -> error_budget.ranked. A hand-edited
disagreement must fail with a message naming both values.
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import error_budget


class TestAfbgMetricsTopRanks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = json.loads((REPO / "docs" / "model_facts.json").read_text())
        cls.metrics = json.loads((REPO / "experiments" / "gpu-sim-afbg" / "metrics.json").read_text())
        cls.ranked = cls.facts["error_budget"]["ranked"]

    def test_top_ranks_matches_computed(self):
        cells = self.metrics["new_cells"]
        stated = self.metrics["top_ranks"]
        computed = error_budget.top_ranks_from_ranked(self.ranked, cells)
        for cell in cells:
            s = stated.get(cell)
            c = computed.get(cell)
            self.assertEqual(
                s, c,
                f"metrics top_ranks[{cell}]={s!r} != computed {c!r}",
            )

    def test_top_ranks_covers_all_new_cells(self):
        self.assertEqual(set(self.metrics["top_ranks"].keys()), set(self.metrics["new_cells"]))


if __name__ == "__main__":
    unittest.main()
