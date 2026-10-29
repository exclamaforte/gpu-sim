"""Bottom link for gpu-sim-7uz2: raw measurement -> anchored facts.

Parses experiments/gpu-sim-7uz2/act_curve_raw.json and recomputes act_factor
from the raw bytes (peak_max - mem_params_grads), NOT from any precomputed
field. Asserts the recomputed value equals what is anchored in model_facts
and metrics.json. A planted edit to the raw file must turn this red.
"""
import json
import statistics
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _dtype_bytes(dtype):
    return 2 if "bfloat" in dtype or "bf16" in dtype else 4


def _recompute(raw):
    cfg = raw["config"]
    dtype_bytes = _dtype_bytes(cfg["dtype"])
    d = cfg["d"]
    d_ff = cfg["d_ff"]
    n_layers = cfg["n_layers"]
    batch = cfg["batch"]
    per_token_width = 3 * d + d_ff
    out = {}
    for r in raw["results"]:
        seq = r["seq"]
        afs = []
        for rep in r["reps"]:
            if "error" in rep:
                continue
            peak = rep["peak_max_bytes"]
            base = rep["mem_params_grads_bytes"]
            delta = peak - base
            bptl = delta / (batch * seq * n_layers)
            af = bptl / (dtype_bytes * per_token_width)
            afs.append(round(af, 4))
        out[str(seq)] = statistics.median(afs) if afs else None
    return out


class Test7uz2RawToFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads((REPO / "experiments" / "gpu-sim-7uz2" / "act_curve_raw.json").read_text())
        cls.facts = json.loads((REPO / "docs" / "model_facts.json").read_text())
        cls.metrics = json.loads((REPO / "experiments" / "gpu-sim-7uz2" / "metrics.json").read_text())
        cls.recomputed = _recompute(cls.raw)
        cls.mac = cls.facts["scaleout_opt"]["measured_act_curve"]

    def test_recomputed_matches_facts(self):
        for seq, val in self.recomputed.items():
            anchored = self.mac["curve"][seq]["act_factor_median"]
            self.assertAlmostEqual(val, anchored, places=3,
                                   msg=f"seq {seq}: raw recompute {val} != anchored {anchored}")

    def test_recomputed_matches_metrics(self):
        for seq, val in self.recomputed.items():
            anchored = self.metrics["curve_act_factor_by_seq"][seq]
            self.assertAlmostEqual(val, anchored, places=3,
                                   msg=f"seq {seq}: raw recompute {val} != metrics {anchored}")

    def test_all_seqs_present(self):
        self.assertEqual(set(self.recomputed.keys()),
                         {str(s) for s in self.raw["config"]["seqs"].split(",")})


if __name__ == "__main__":
    unittest.main()
