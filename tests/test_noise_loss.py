#!/usr/bin/env python3
"""Tests for 9.3 noise-aware loss (CPU). Contracts pinned here:

- shared gaussian_nll_with_floor == experiment nll_floor (no drift)
- proper scoring: loss minimized at the true variance (synthetic cells)
- floor caps variance collapse (with sigma bounded, without explodes)
- DualHeadPredictor emits VARIANCE (positive), not log-variance
- misparameterization (log-var fed as var) shifts the optimum, proving
  the contract test is load-bearing, not vacuous
- TrainingLoop dual branch delegates to the shared loss + sigma plumbing
"""
import math
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from epic_u2o.model import (  # noqa: E402
    DualHeadPredictor,
    ModelConfig,
    create_model,
    gaussian_nll_with_floor,
)
from epic_u2o.train_trans_91 import nll_floor  # noqa: E402
from epic_u2o.training_loop import (  # noqa: E402
    TrainingLoop,
    _batch_sigma,
)


class TestSharedLoss(unittest.TestCase):
    def test_matches_experiment_alias(self):
        g = torch.Generator().manual_seed(0)
        mu = torch.randn(16, 1, generator=g)
        vp = torch.rand(16, 1, generator=g) + 0.05
        y = torch.randn(16, 1, generator=g)
        s = torch.rand(16, 1, generator=g) * 0.3
        a = gaussian_nll_with_floor(mu, vp, y, s)
        b = nll_floor(mu, vp, y, s)
        self.assertAlmostEqual(a.item(), b.item(), places=6)

    def test_proper_scoring_min_at_truth(self):
        # cells ~ N(0, V*), V* = 0.25: loss over candidate var minimized at V*
        g = torch.Generator().manual_seed(1)
        y = torch.randn(4000, 1, generator=g) * 0.5
        mu = torch.zeros_like(y)
        cands = [0.05, 0.1, 0.25, 0.5, 1.0]
        losses = [gaussian_nll_with_floor(mu, torch.full_like(y, c), y, 0.0).item()
                  for c in cands]
        self.assertEqual(cands[losses.index(min(losses))], 0.25)

    def test_floor_caps_collapse(self):
        mu = torch.zeros(8, 1)
        y = torch.ones(8, 1) * 0.5
        tiny = torch.full((8, 1), 1e-9)
        sig = torch.full((8, 1), 0.2)
        with_floor = gaussian_nll_with_floor(mu, tiny, y, sig).item()
        self.assertTrue(math.isfinite(with_floor) and with_floor < 10.0)
        no_floor = gaussian_nll_with_floor(mu, tiny, y, 0.0).item()
        self.assertGreater(no_floor, with_floor + 5.0)

    def test_misparameterization_shifts_optimum(self):
        # negative control: feeding LOG-variance numbers (in (-2, 0.7)) into
        # the variance slot must NOT agree with the correct call -- else the
        # contract test would be vacuous.
        g = torch.Generator().manual_seed(2)
        mu = torch.zeros(64, 1)
        y = torch.randn(64, 1, generator=g) * 0.5
        logv = torch.full((64, 1), math.log(0.25))  # correct log-var
        wrong = gaussian_nll_with_floor(mu, logv, y, 0.0).item()
        right = gaussian_nll_with_floor(mu, torch.full((64, 1), 0.25), y,
                                        0.0).item()
        self.assertNotAlmostEqual(wrong, right, places=3)


class TestHeadContract(unittest.TestCase):
    def test_head_emits_positive_variance(self):
        cfg = ModelConfig(vocab_size=64, max_seq_len=16, d_model=32,
                          n_heads=4, n_layers=1, d_ff=64, numeric_dim=33,
                          use_numeric=True)
        m = DualHeadPredictor(cfg)
        m.eval()
        ids = torch.randint(1, 64, (4, 16))
        mask = torch.ones(4, 16, dtype=torch.long)
        num = torch.randn(4, 33)
        with torch.no_grad():
            mean, var = m(ids, mask, num)
        self.assertEqual(mean.shape, (4, 1))
        self.assertTrue(bool((var > 0).all()),
                        "second head must emit positive variance")

    def test_create_model_dual_head_same_contract(self):
        m = create_model(vocab_size=64, max_seq_len=16, d_model=32, n_heads=4,
                         n_layers=1, d_ff=64, numeric_dim=33, use_numeric=True,
                         dual_head=True)
        m.eval()
        with torch.no_grad():
            _, var = m(torch.randint(1, 64, (2, 16)),
                       torch.ones(2, 16, dtype=torch.long),
                       torch.randn(2, 33))
        self.assertTrue(bool((var > 0).all()))


class TestLoopDelegation(unittest.TestCase):
    def test_loop_delegates_to_shared(self):
        g = torch.Generator().manual_seed(3)
        mu = torch.randn(8, 1, generator=g)
        vp = torch.rand(8, 1, generator=g) + 0.05
        y = torch.randn(8, 1, generator=g)
        s = torch.rand(8, 1, generator=g) * 0.2
        got = TrainingLoop._gaussian_nll_loss(None, mu, vp, y, s)
        want = gaussian_nll_with_floor(mu, vp, y, s)
        self.assertAlmostEqual(got.item(), want.item(), places=6)

    def test_batch_sigma_defaults_and_passthrough(self):
        self.assertEqual(_batch_sigma({}, "cpu"), 0.0)
        s = torch.tensor([0.1, 0.2])
        self.assertTrue(torch.equal(_batch_sigma({"sigma": s}, "cpu"), s))


if __name__ == "__main__":
    unittest.main(verbosity=2)
