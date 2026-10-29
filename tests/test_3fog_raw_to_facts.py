"""Bottom link for gpu-sim-3fog: raw measurement -> anchored facts."""
import json
import statistics
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class Test3fogRawToFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((REPO / "experiments" / "gpu-sim-3fog" / "bw_raw.json").read_text())
        cls.facts = json.loads((REPO / "docs" / "model_facts.json").read_text())
        cls.metrics = json.loads((REPO / "experiments" / "gpu-sim-3fog" / "metrics.json").read_text())

    def test_recomputed_achieved_matches_facts(self):
        copy_med = self.raw["results"]["copy"]["median_bytes_per_s"]
        spec = self.raw["spec_bytes_per_s"]
        recomputed = copy_med / spec
        anchored = self.facts["achieved_bw_measured"]["achieved_bw_scale"]
        self.assertAlmostEqual(recomputed, anchored, places=6)

    def test_recomputed_achieved_matches_metrics(self):
        copy_med = self.raw["results"]["copy"]["median_bytes_per_s"]
        spec = self.raw["spec_bytes_per_s"]
        recomputed = copy_med / spec
        self.assertAlmostEqual(recomputed, self.metrics["achieved_bw_scale"], places=6)


if __name__ == "__main__":
    unittest.main()
