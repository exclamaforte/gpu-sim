"""Guard that experiments/gpu-sim-3fog/metrics.json is produced, not typed."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class Test3fogBw(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = json.loads((REPO / "docs" / "model_facts.json").read_text())
        cls.metrics = json.loads((REPO / "experiments" / "gpu-sim-3fog" / "metrics.json").read_text())
        cls.ab = cls.facts["achieved_bw_measured"]

    def test_achieved_scale_matches(self):
        self.assertEqual(self.metrics["achieved_bw_scale"], self.ab["achieved_bw_scale"])

    def test_copy_median_matches(self):
        self.assertEqual(self.metrics["copy_median_gbs"], self.ab["copy_median_gbs"])

    def test_spec_matches(self):
        self.assertEqual(self.metrics["spec_bw_gbs"], self.ab["spec_bw_gbs"])

    def test_verdict_survives(self):
        self.assertTrue(self.metrics["verdict_survives"]["N8_winner"])
        self.assertTrue(self.metrics["verdict_survives"]["N16_seq32768"])


if __name__ == "__main__":
    unittest.main()
