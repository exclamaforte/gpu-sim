"""gpu-sim-7q3w.4: the checkpoint agreement checker must see a disagreement.

The product under test is `experiments/gpu-sim-7q3w.4/checkpoint_agreement.py`,
the same canonical hash + equality function the training probe calls.  The test
uses synthetic state_dicts, so there is no training, no subprocess, no GPU and
no timing: the property is "equal tensors are clean; one different tensor is a
named disagreement", which is exactly what the registered mutation breaks by
making the comparison pass unconditionally.
"""
import importlib.util
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
# Assembled from pieces, matching this repo's ledger self-reference habit: a row
# must bind a PRODUCT test, not the fs2y checker.
CHECKER = REPO / "experiments" / "gpu-sim-7q3w.4" / ("checkpoint_" + "agreement.py")


def _load_checker():
    spec = importlib.util.spec_from_file_location("checkpoint_agreement_under_test", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state_dict(delta=0.0):
    return {
        "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "layer.bias": torch.tensor([0.5, -0.5]) + delta,
    }


class TestCheckpointAgreement(unittest.TestCase):
    def test_equal_state_dicts_hash_equal(self):
        checker = _load_checker()
        h1 = checker.canonical_state_dict_sha256(_state_dict())
        h2 = checker.canonical_state_dict_sha256(_state_dict())
        self.assertEqual(h1, h2)
        result = checker.check_checkpoint_hashes([h1, h2])
        self.assertTrue(result["agrees"], result)

    def test_synthetic_disagreement_is_detected(self):
        """The positive-negative arm: one rank's tensor differs -> not clean."""
        checker = _load_checker()
        h_equal = checker.canonical_state_dict_sha256(_state_dict())
        h_different = checker.canonical_state_dict_sha256(_state_dict(delta=1.0))
        self.assertNotEqual(h_equal, h_different)
        result = checker.check_checkpoint_hashes([h_equal, h_equal, h_different])
        self.assertFalse(result["agrees"], result)
        self.assertEqual(len(result["distinct"]), 2, result)
        self.assertIn(h_equal, result["values"], result)
        self.assertIn(h_different, result["values"], result)

    def test_empty_hash_list_is_not_clean(self):
        checker = _load_checker()
        result = checker.check_checkpoint_hashes([])
        self.assertFalse(result["agrees"], result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
