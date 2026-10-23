#!/usr/bin/env python3
"""9.6 proof (CPU): the SHARED TrainingLoop trains on REAL data.

Audit demand: the loop had only ever seen random tensors. Here it trains a
tiny dual-head model on 9.1's real 456-row corpus (family_holdout_v1 split),
with 6.6 sigma floors flowing, eval hooks firing, best-tracking, and a
checkpoint resume round-trip. Runs CPU in ~1 min; also serves as run.sh.
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_trans_91 as T  # noqa: E402
from epic_u2o.model import create_model  # noqa: E402
from epic_u2o.training_loop import (  # noqa: E402
    TrainingLoop,
    TrainingConfig,
    _batch_sigma,
)

EXP = REPO / "experiments" / "gpu-sim-u2o.9.6"
PROOF = EXP / "_proof"


def _loaders():
    rows = [json.loads(l) for l in
            (REPO / "experiments/gpu-sim-u2o.9.1"
             / "train_data_91.jsonl").read_text().splitlines()]
    parts, _, _ = T.family_holdout_split(rows, REPO)
    tr_ds = T.TTDataset(parts["train"])
    va_ds = T.TTDataset(parts["val"], tr_ds.mean, tr_ds.std, tr_ds.mask)
    te_ds = T.TTDataset(parts["test"], tr_ds.mean, tr_ds.std, tr_ds.mask)
    kw = dict(batch_size=32, num_workers=0)
    return parts, (DataLoader(tr_ds, shuffle=True, drop_last=True, **kw),
                   DataLoader(va_ds, **kw), DataLoader(te_ds, **kw))


def _tiny_model():
    torch.manual_seed(0)
    return create_model(vocab_size=256, max_seq_len=T.SEQ, d_model=64,
                        n_heads=4, n_layers=1, d_ff=128, numeric_dim=33,
                        use_numeric=True, dual_head=True, fusion_type="concat")


def _cfg(ckpt_dir, log_dir):
    return TrainingConfig(device="cpu", max_steps=30, eval_every_steps=10,
                          eval_batches=2, log_every_steps=10, keep_last_n=2,
                          checkpoint_dir=str(ckpt_dir), log_dir=str(log_dir),
                          use_amp=False)


class TestLoopOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts, (cls.tr, cls.va, cls.te) = _loaders()
        assert (len(cls.tr.dataset), len(cls.va.dataset),
                len(cls.te.dataset)) == (342, 57, 57)

    def test_sigma_floor_engaged_on_real_batch(self):
        b = next(iter(self.tr))
        s = _batch_sigma(b, "cpu")
        self.assertTrue(torch.is_tensor(s))
        self.assertGreater(float(s.max()), 0.0)

    def test_train_eval_checkpoint_resume(self):
        ck, lg = PROOF / "ckpts", PROOF / "logs"
        for d in (ck, lg):
            d.mkdir(parents=True, exist_ok=True)
        for f in list(ck.glob("*.pt")):
            f.unlink()
        hook_calls = []
        loop = TrainingLoop(_tiny_model(), _cfg(ck, lg),
                            train_loader=self.tr, val_loader=self.va,
                            eval_fn=lambda m, s, v: hook_calls.append(s))
        res = loop.run()
        self.assertEqual(res["total_steps"], 30)
        # loss decreases (half-means, tolerant of noise)
        tl = loop.train_losses
        self.assertLess(float(np.mean(tl[len(tl) // 2:])),
                        float(np.mean(tl[:len(tl) // 2])))
        # frequent eval hooks: steps 10/20/30
        self.assertEqual(len(loop.val_losses), 3)
        self.assertEqual(hook_calls, [10, 20, 30])
        for v in loop.val_losses:
            self.assertTrue(np.isfinite(v))
        # checkpointing: best + bounded rotation
        self.assertTrue((ck / "best_model.pt").exists())
        self.assertLessEqual(len(list(ck.glob("checkpoint_step_*.pt"))), 3)
        # resume round-trip into a fresh loop
        loop2 = TrainingLoop(_tiny_model(), _cfg(ck, lg),
                             train_loader=self.tr, val_loader=self.va)
        loop2.load_checkpoint(str(ck / "best_model.pt"))
        self.assertEqual(loop2.step, 30)
        for p, q in zip(loop.model.parameters(), loop2.model.parameters()):
            self.assertTrue(torch.equal(p.cpu(), q.cpu()))
        # final frame: test-set report keys
        rep = loop.evaluate(self.te, max_batches=2)
        for k in ("loss", "mape", "mse", "r2"):
            self.assertIn(k, rep)
        metrics = {"bead": "gpu-sim-u2o.9.6", "split_sizes": [342, 57, 57],
                   "total_steps": res["total_steps"],
                   "train_loss_first": round(float(tl[0]), 4),
                   "train_loss_last": round(float(tl[-1]), 4),
                   "val_losses": [round(float(v), 4)
                                  for v in loop.val_losses],
                   "best_val_loss": round(float(res["best_val_loss"]), 4),
                   "eval_hook_steps": hook_calls,
                   "test_report_keys": sorted(rep.keys()),
                   "resume_step_restored": loop2.step}
        (EXP / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print("\n9.6 metrics:", json.dumps(metrics))


if __name__ == "__main__":
    unittest.main(verbosity=2)
