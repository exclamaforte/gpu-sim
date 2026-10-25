#!/usr/bin/env python3
"""Tests for 11.1 Step-1 overfit sanity (CPU). A tiny dual-head model must
memorize 32 fixed REAL rows (train loss -> ~0), proving the
tokenize->batch->model->loss pipeline is functional end to end. LR=0
control proves the decrease is learning, not bookkeeping.
"""
import json
import signal
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))

import os  # noqa: E402
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch  # noqa: E402
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torch.utils.data import DataLoader  # noqa: E402

import train_trans_91 as T  # noqa: E402
from epic_u2o.model import create_model, gaussian_nll_with_floor  # noqa: E402

EXP = REPO / "experiments" / "gpu-sim-u2o.11.1"
N_FIT, STEPS, CTRL_STEPS = 32, 35, 5


def _subset():
    rows = [json.loads(l) for l in
            (REPO / "experiments/gpu-sim-u2o.9.1"
             / "train_data_91.jsonl").read_text().splitlines()]
    parts, _, _ = T.family_holdout_split(rows, REPO)
    sub = parts["train"][:N_FIT]
    assert len({r["y"] for r in sub}) > 1, "degenerate targets"
    ds = T.TTDataset(sub)
    # full-batch: every step sees the same 32 rows, so first-vs-last
    # compares the same batch (the LR=0 control needs exact equality).
    return DataLoader(ds, batch_size=N_FIT, shuffle=False, num_workers=0)


def _model():
    torch.manual_seed(0)
    return create_model(vocab_size=256, max_seq_len=T.SEQ, d_model=64,
                        n_heads=4, n_layers=1, d_ff=128, numeric_dim=33,
                        use_numeric=True, dual_head=True, fusion_type="concat")


def _run(loader, lr, steps, train_mode=True):
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    if train_mode:
        m.train()
    else:
        m.eval()  # control: no dropout, frozen net => bit-identical losses
    first, last = None, None
    for _ in range(steps):
        for b in loader:
            mu, vp = m(b["input_ids"], b["attention_mask"],
                       b["numeric_features"])
            loss = gaussian_nll_with_floor(mu, vp, b["target"], b["sigma"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = loss.item()
            if first is None:
                first = last
    return first, last


class TestOverfitSanity(unittest.TestCase):
    def _timeout_handler(self, signum, frame):
        raise TimeoutError("test_overfit_111 exceeded 120s wall budget")

    def test_overfits_fixed_real_subset(self):
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(120)
        loader = _subset()
        first, last = _run(loader, 3e-4, STEPS)
        self.assertLess(last, 0.05 * first)
        ctrl = DataLoader(loader.dataset, batch_size=N_FIT,
                          shuffle=False, num_workers=0)
        c_first, c_last = _run(ctrl, 0.0, CTRL_STEPS, train_mode=False)
        self.assertAlmostEqual(c_first, c_last, places=6)
        metrics = {"bead": "gpu-sim-u2o.11.1", "n_fit": N_FIT,
                   "steps": STEPS, "model": "d64/4h/1L dual-head concat",
                   "lr3e-4_first": round(first, 4),
                   "lr3e-4_last": round(last, 4),
                   "lr0_first": round(c_first, 4),
                   "lr0_last": round(c_last, 4)}
        (EXP / "metrics.json").write_text(json.dumps(metrics, indent=2))
        signal.alarm(0)
        print("\n11.1 metrics:", json.dumps(metrics))


if __name__ == "__main__":
    unittest.main(verbosity=2)
