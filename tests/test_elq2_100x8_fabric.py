"""Guard that the 100x8 fabric-uncertainty sensitivity result is reproducible."""
import json, sys, unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

from scaleout_answers import build_spec, answer, MODELS, CONVENTIONS


FACTORS = [1 / 1.233, 1 / 1.10, 1.0, 1.10, 1.233]


class TestElq2FabricSensitivity(unittest.TestCase):
    def test_flipping_count(self):
        spec = build_spec(100)
        n = 800
        flipped = 0
        for mname, model in MODELS.items():
            for batch in (1, 8):
                for cname, scale in CONVENTIONS.items():
                    winners = {}
                    for f in FACTORS:
                        r = answer(spec, model, n, scale * f, batch)
                        winners[round(f, 6)] = r["config"] if r else None
                    base_w = winners[1.0]
                    if any(w != base_w for f, w in winners.items() if f != 1.0):
                        flipped += 1
        self.assertEqual(flipped, 5,
                         f"expected 5 of 12 rows to flip under fabric band, got {flipped}")

    def test_metrics_matches(self):
        metrics = json.loads((REPO / "experiments" / "gpu-sim-elq2" / "metrics.json").read_text())
        self.assertEqual(metrics["n_rows"], 12)
        self.assertEqual(metrics["n_rows_flipping"], 5)

    def test_fabric_scale_noop_default(self):
        spec = build_spec(100)
        r = answer(spec, MODELS["llama3-8b"], 800, 1.0, 1, 0.0)
        r2 = answer(spec, MODELS["llama3-8b"], 800, 1.0, 1, 0.0, fabric_scale=1.0)
        self.assertEqual(r["config"], r2["config"])
        self.assertEqual(r["margin_pct"], r2["margin_pct"])


if __name__ == "__main__":
    unittest.main()
