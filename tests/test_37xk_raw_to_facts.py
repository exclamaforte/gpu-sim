"""Bottom link for gpu-sim-37xk: raw measurement -> anchored facts."""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


class Test37xkRawToFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((REPO / "experiments" / "gpu-sim-37xk" / "measured_opt_bytes.json").read_text())
        cls.facts = json.loads((REPO / "docs" / "model_facts.json").read_text())
        cls.msb = cls.facts["scaleout_opt"]["measured_style_bytes"]

    def test_n_params_matches(self):
        self.assertEqual(self.msb["n_params"], self.raw["n_params"])

    def test_steps_match(self):
        self.assertEqual(self.msb["steps"], self.raw["steps"])

    def test_bf16_fused_rounds(self):
        self.assertAlmostEqual(self.msb["bf16_moments"]["fused"],
                               round(self.raw["measured_bytes_per_param"]["bf16_moments"]["measured_peak_fused"], 2), places=2)

    def test_fp32_fused_rounds(self):
        self.assertAlmostEqual(self.msb["fp32_moments"]["fused"],
                               round(self.raw["measured_bytes_per_param"]["fp32_moments"]["measured_peak_fused"], 2), places=2)

    def test_params_grads_rounds(self):
        self.assertAlmostEqual(self.msb["bf16_moments"]["params_grads"],
                               round(self.raw["measured_bytes_per_param"]["bf16_moments"]["params_grads"], 2), places=2)


if __name__ == "__main__":
    unittest.main()
