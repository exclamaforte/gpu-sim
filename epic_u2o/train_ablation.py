#!/usr/bin/env python3
"""Minimal ablation training entrypoint (gpu-sim-u2o.8.4).

Trains ONE representation arm with the shared tiny encoder; representation
is the only input difference (matched-size by construction: identical
model, hparams, train rows, seed).

  python -m epic_u2o.train_ablation --arm source|ptx|hybrid \
      --data experiments/gpu-sim-u2o.8.4/ablation_data.jsonl \
      --out /abs/path/arm_dir [--smoke]

Arm inputs (max 512 ids each): source = src_ids; ptx = ptx_ids;
hybrid = src_ids[:256] + ptx_ids[:256]. use_numeric=False for all arms
(the batch carries a dummy numeric vector for TrainingLoop's contract;
the model ignores it).

Also closes the doc gap: representation_ablation.md named
`python -m epic_u2o.train`, which never existed; this module is the real
entrypoint (doc updated alongside).
"""
import argparse
import json
import math
import random
import sys
# Several epic_u2o modules hardcode sys.path.insert(0, main-tree); scrub it
# so epic_u2o.* imports resolve in the LAUNCH tree (worktree runs would
# otherwise silently mix worktree + main-tree modules).
sys.path = [p for p in sys.path if p != "/home/gabe/Documents/gpu-sim"]
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from epic_u2o.model import create_model
from epic_u2o.training_loop import TrainingConfig, TrainingLoop
# N.B. generalization_suites hardcodes sys.path.insert(0, main-tree) at
# import; it MUST come after the imports above, else model/training_loop
# silently resolve to the main tree. structural_split binds main-tree
# (byte-identical, pure) -- tolerated, noted.
from epic_u2o.generalization_suites import build_suites

SEQ = 512
HYB_SPLIT = 256
SEED = 0
EPOCHS = 30
BATCH = 64
LR = 3e-4


class AblationDataset(Dataset):
    def __init__(self, rows, arm):
        self.rows = rows
        self.arm = arm

    def __len__(self):
        return len(self.rows)

    def _ids(self, r):
        if self.arm == "source":
            return r["src_ids"][:SEQ]
        if self.arm == "ptx":
            return r["ptx_ids"][:SEQ]
        return (r["src_ids"][:HYB_SPLIT] + r["ptx_ids"][:HYB_SPLIT])[:SEQ]

    def __getitem__(self, idx):
        r = self.rows[idx]
        ids = self._ids(r)[:SEQ]
        mask = [1] * len(ids)
        pad = SEQ - len(ids)
        ids = ids + [0] * pad
        mask = mask + [0] * pad
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                # Contract filler: model runs use_numeric=False, ignored.
                "numeric_features": torch.zeros(1, dtype=torch.float),
                "target": torch.tensor([r["log_target"]], dtype=torch.float),
                "kernel_hash": r["kernel_hash"]}


def mape_on(model, device, rows, arm, batch_size=256):
    model.eval()
    ds = AblationDataset(rows, arm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    errs, preds = [], {}
    with torch.no_grad():
        for b in loader:
            out = model(b["input_ids"].to(device),
                        b["attention_mask"].to(device),
                        b["numeric_features"].to(device))
            for h, lp in zip(b["kernel_hash"],
                             out.squeeze(-1).cpu().tolist()):
                preds[h] = math.exp(lp)
    for r in rows:
        errs.append(abs(preds[r["kernel_hash"]] - r["true_ms"]) / r["true_ms"])
    return sum(errs) / len(errs), preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["source", "ptx", "hybrid"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()]
    suites = build_suites(rows, seed=42)
    train_rows = suites["random"]["train"]
    val_rows = suites["random"]["val"]
    if args.smoke:
        train_rows = train_rows[:64]
        val_rows = val_rows[:16]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(vocab_size=256, max_seq_len=SEQ, d_model=64,
                         n_heads=4, n_layers=2, d_ff=256, use_numeric=False,
                         dual_head=False)
    n_params = sum(p.numel() for p in model.parameters())
    train_loader = DataLoader(AblationDataset(train_rows, args.arm),
                              batch_size=BATCH, shuffle=True, num_workers=0,
                              pin_memory=(device == "cuda"), drop_last=True)
    val_loader = DataLoader(AblationDataset(val_rows, args.arm),
                            batch_size=BATCH, shuffle=False, num_workers=0)
    steps_per_epoch = max(1, len(train_loader))
    cfg = TrainingConfig(lr=LR, weight_decay=0.01, max_steps=steps_per_epoch * EPOCHS,
                         eval_every_steps=max(1, steps_per_epoch // 2),
                         eval_batches=100, log_every_steps=max(1, steps_per_epoch // 4),
                         checkpoint_dir=str(out / "ckpt"), log_dir=str(out / "logs"),
                         use_amp=(device == "cuda"), device=device)
    loop = TrainingLoop(model=model, config=cfg)
    loop.run(train_loader, val_loader, max_epochs=EPOCHS)

    # Best checkpoint for the verdict (val-loss selection done by the loop).
    best = out / "ckpt" / "best_model.pt"
    if best.exists():
        model.load_state_dict(torch.load(str(best), map_location=device)["model_state_dict"])
    model.to(device)
    suite_mape, all_preds = {}, {}
    for name in ("unseen_families", "unseen_problem_sizes", "unseen_occupancy", "random"):
        test = suites[name]["test"]
        m, preds = mape_on(model, device, test, args.arm)
        suite_mape[name] = {"mape": m, "n": len(test)}
        for h, pm in preds.items():
            all_preds.setdefault(h, {}).update({f"pred_{name}": pm})
    train_mape, _ = mape_on(model, device, train_rows, args.arm)
    arm_metrics = {"arm": args.arm, "n_params": n_params, "seed": SEED,
                   "epochs": EPOCHS, "batch": BATCH, "lr": LR,
                   "n_train": len(train_rows),
                   "train_mape": train_mape, "suites": suite_mape,
                   "device": device}
    (out / "arm_metrics.json").write_text(json.dumps(arm_metrics, indent=2))
    with (out / "predictions.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({"kernel_hash": r["kernel_hash"],
                                 "true_ms": r["true_ms"],
                                 **all_preds.get(r["kernel_hash"], {})}) + "\n")
    print(f"arm {args.arm}: params={n_params} train_mape={train_mape:.4f} " +
          " ".join(f"{k}={v['mape']:.4f}(n={v['n']})"
                   for k, v in suite_mape.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
