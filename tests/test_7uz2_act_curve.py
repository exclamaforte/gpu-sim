"""Guard that experiments/gpu-sim-7uz2/metrics.json is produced, not typed."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class Test7uz2ActCurve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads((REPO / "experiments" / "gpu-sim-7uz2" / "metrics.json").read_text())
        cls.facts = json.loads((REPO / "docs" / "model_facts.json").read_text())
        cls.mac = cls.facts["scaleout_opt"]["measured_act_curve"]

    def test_curve_matches_facts(self):
        self.assertEqual(self.metrics["curve_act_factor_by_seq"],
                         {str(k): v["act_factor_median"] for k, v in self.mac["curve"].items()})

    def test_spread_zero_at_fixed_length(self):
        for v in self.metrics["spread_at_fixed_length"].values():
            self.assertEqual(v, 0.0)

    def test_seq_8192_below_flip(self):
        self.assertTrue(self.metrics["verdict"]["seq_8192_below_flip"])

    def test_seq_32768_above_flip(self):
        self.assertTrue(self.metrics["verdict"]["seq_32768_above_flip"])

    def test_five_seq_lengths(self):
        self.assertEqual(len(self.metrics["seqs"]), 5)


if __name__ == "__main__":
    unittest.main()
