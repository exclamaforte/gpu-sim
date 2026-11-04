import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "experiments" / "gpu-sim-nlgf" / "metrics_power.json"


class TestNlgfPower(unittest.TestCase):
    def test_metrics_has_required_fields(self):
        data = json.loads(METRICS.read_text())
        for key in ("n_cells_total", "n_usable", "paired_delta_ci", "label", "achieved_mde",
                    "n_units", "unit_rule", "label_scope", "model_verdict", "per_config_usable"):
            self.assertIn(key, data)
        assert data["n_cells_total"] == 56
        assert data["n_replicates_per_cell"] == 3
        assert data["n_units"] == 1
        assert data["unit_rule"] is not None
        assert data["label_scope"] == "single configuration; protocol demonstration, not a model verdict"
        assert data["model_verdict"] == "INSUFFICIENT"
        assert data["label"] in ("BETTER", "WORSE", "TIE", "UNRESOLVED", "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
