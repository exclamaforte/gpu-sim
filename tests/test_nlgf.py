import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "experiments" / "gpu-sim-u5r4" / "metrics_power.json"


class TestNlgfPower(unittest.TestCase):
    def test_metrics_has_required_fields(self):
        data = json.loads(METRICS.read_text())
        for key in ("n_cells_total", "n_usable", "paired_delta_ci", "label", "achieved_mde"):
            self.assertIn(key, data)
        assert data["n_cells_total"] == 56
        assert data["n_replicates_per_cell"] == 3
        assert data["label"] in ("BETTER", "WORSE", "TIE", "UNRESOLVED", "INSUFFICIENT")


if __name__ == "__main__":
    unittest.main()
