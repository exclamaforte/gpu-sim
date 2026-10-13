#!/usr/bin/env python3
"""Tests for the 9.1 transformer baseline. Pins acceptance 1-5 + 9."""
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "gpu-sim-u2o.9.1"
sys.path.insert(0, str(REPO / "epic_u2o"))

import train_trans_91 as T  # noqa: E402


def train_rows():
    p = EXP / "train_data_91.jsonl"
    if not p.exists():
        raise unittest.SkipTest("run build_data_91.py first")
    return [json.loads(l) for l in p.read_text().splitlines()]


class TestNLL(unittest.TestCase):
    def test_round_trip_synthetic_cell(self):
        import torch
        mu = torch.tensor([[1.0]])
        var_p = torch.tensor([[0.25]])
        y = torch.tensor([[1.0]])
        sigma = torch.tensor([[0.1]])
        got = T.nll_floor(mu, var_p, y, sigma).item()
        self.assertAlmostEqual(got, 0.5 * math.log(0.26), places=6)

    def test_sigma_floor_consumed_not_learned(self):
        import torch
        mu = torch.tensor([[0.0]])
        y = torch.tensor([[1.0]])
        small = T.nll_floor(mu, torch.tensor([[1e-9]]),
                            y, torch.tensor([[0.5]])).item()
        nosig = T.nll_floor(mu, torch.tensor([[1e-9]]),
                            y, torch.tensor([[0.0]])).item()
        self.assertLess(small, nosig,
                        "floor must cap the loss when var_pred -> 0")


class TestProvenance(unittest.TestCase):
    def test_dataset_keys_are_model_inputs_only(self):
        rows = train_rows()
        item = T.TTDataset(rows[:8]).__getitem__(0)
        self.assertEqual(set(item.keys()),
                         {"input_ids", "attention_mask", "numeric_features",
                          "target", "sigma", "kernel_hash"})
        self.assertIsInstance(item["kernel_hash"], str)

    def test_no_hash_derived_values(self):
        rows = train_rows()
        for r in rows[:16]:
            h = hashlib.sha256(r["hash"].encode()).hexdigest()
            for v in r["src_ids"][:8]:
                self.assertNotEqual(str(v), h[:8])
            self.assertNotIn("op_type", r)
            self.assertNotIn("model_bias", r)

    def test_op_type_never_an_input(self):
        rows = train_rows()
        item = T.TTDataset(rows[:4]).__getitem__(0)
        self.assertNotIn("op_type", item)
        self.assertNotIn("family", " ".join(item.keys()))


class TestOOB(unittest.TestCase):
    def test_known_oob_absent_from_join(self):
        rows = train_rows()
        oob = json.loads((REPO / "experiments/gpu-sim-0vk/oob_frozen.json")
                         .read_text())
        oob_hashes = {r["kernel_hash"] for r in oob["rows"]}
        self.assertEqual(len(oob_hashes), 4)
        overlap = oob_hashes & {r["hash"] for r in rows}
        self.assertEqual(overlap, set())


class TestNegCtrl(unittest.TestCase):
    def test_overfit_and_flag(self):
        p = EXP / "negctrl.json"
        if not p.exists():
            raise unittest.SkipTest("run negctrl_91.py first")
        n = json.loads(p.read_text())
        self.assertTrue(n["collapsed_flag"])
        self.assertLess(n["train_mean_abs"], n["heldout_mean_abs"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
