"""Guard that experiments/gpu-sim-3fog/metrics.json carries all runs (produced)."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class Test3fogProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads((REPO / "experiments" / "gpu-sim-3fog" / "metrics.json").read_text())

    def test_runs_all_present(self):
        self.assertIn("runs_all", self.metrics)
        self.assertGreaterEqual(len(self.metrics["runs_all"]), 2)

    def test_anchored_run_present(self):
        anchored = [r for r in self.metrics["runs_all"] if r.get("anchored")]
        self.assertEqual(len(anchored), 1)
        self.assertAlmostEqual(anchored[0]["achieved_bw_scale"], self.metrics["achieved_bw_scale"], places=6)

    def test_contention_note_present(self):
        self.assertIn("contention_finding_corrected", self.metrics)
        self.assertTrue(self.metrics["contention_finding_corrected"])

    def test_every_run_has_job_id(self):
        for r in self.metrics["runs_all"]:
            self.assertIn("job_id", r)


if __name__ == "__main__":
    unittest.main()
