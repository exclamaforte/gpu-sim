#!/usr/bin/env python3
"""Tests for 9.2 fusion wiring. The numeric_dim test pins a real clobber:
threading fusion_type through create_model once dropped the numeric_dim
line (26-dim encoder, 33-dim inputs) -- caught by smoke before GPU."""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.model import create_model  # noqa: E402


class TestFusionWiring(unittest.TestCase):
    def test_numeric_dim_survives_fusion_kwargs(self):
        for f in ("concat", "cross_attention"):
            m = create_model(d_model=64, n_heads=4, n_layers=1, d_ff=128,
                             numeric_dim=33, use_numeric=True, dual_head=True,
                             fusion_type=f)
            lin = m.numeric_encoder.net[0]
            self.assertEqual(lin.in_features, 33, f)

    def test_unknown_fusion_rejected(self):
        with self.assertRaises(ValueError):
            create_model(d_model=32, n_heads=2, n_layers=1, d_ff=64,
                         numeric_dim=33, use_numeric=True, dual_head=True,
                         fusion_type="additive")

    def test_full_dims_survive(self):
        """The dims-clobber: threading fusion once dropped n_heads /
        n_layers / d_ff, silently building 6-layer 20M models from 12-layer
        code (caught by param census, not by tests). Never again."""
        m = create_model(vocab_size=256, max_seq_len=512, d_model=512,
                         n_heads=8, n_layers=12, d_ff=2048, numeric_dim=33,
                         use_numeric=True, dual_head=True,
                         fusion_type="concat")
        self.assertEqual(len(m.encoder.layers), 12)
        self.assertEqual(sum(p.numel() for p in m.parameters()), 38950658)


if __name__ == "__main__":
    unittest.main(verbosity=2)
