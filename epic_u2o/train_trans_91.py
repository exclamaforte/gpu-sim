#!/usr/bin/env python3
"""gpu-sim-u2o.9.1 transformer training entrypoint (Stage 1 scratch + Stage 2).

Dual-head encoder (mean, var) + Gaussian NLL with the 6.6 per-cell noise
floor ADDED (var_total = var_pred + sigma_cell^2): the floor is consumed,
never learned away, and model_bias stays out (acceptance 4/5).

Naming note (not fixed here -- 9.3 owns loss semantics): DualHeadPredictor's
second head applies softplus, so it emits a VARIANCE despite the
`log_var_head` name. This script treats it as var_pred and documents that.

Inputs per row: src_ids (8.1 source tokens, cap 512) + num33 z-scored with
TRAIN-fitted mean/std (recorded). op_type/family metadata are split keys
only; kernel_hash never enters the pipeline (acceptance 2/3, tested).

  python -m epic_u2o.train_trans_91 --mode scratch --out <dir>   # CPU probe
  python -m epic_u2o.train_trans_91 --mode full --data ... --out <dir> [--resume CKPT]
"""
import argparse
import json
import math
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from epic_u2o.model import create_model

SEQ = 512
SEED = 0
EPOCHS = 20
BATCH = 32
LR = 3e-4
# Stage-2 encoder (~cclm 48M class; exact n_params recorded per run).
D_MODEL, N_HEADS, N_LAYERS, D_FF = 512, 8, 12, 2048


def _sigusr1_exit(signum, frame):
    print("PREEMPTED-SIGUSR1", flush=True)
    raise SystemExit(75)


signal.signal(signal.SIGUSR1, _sigusr1_exit)


def nll_floor(mu, var_p, y, sigma):
    """Gaussian NLL with measurement floor added. All tensors [B,1]."""
    var_t = var_p + sigma * sigma
    return 0.5 * (torch.log(var_t) + (y - mu).pow(2) / var_t).mean()


class TTDataset(Dataset):
    def __init__(self, rows, mean=None, std=None, mask=None):
        self.rows = rows
        x = np.array([r["num33"] for r in rows], dtype=np.float64)
        self.mean = mean if mean is not None else x.mean(axis=0)
        raw_std = x.std(axis=0)
        # Constant features (16/33 have train std exactly 0) carry no
        # signal; dividing by (0+eps) maps a test-time differing constant
        # to ~1e9 and detonates the model (caught: test err 1e14). Mask.
        self.mask = (mask if mask is not None
                     else (raw_std > 1e-12)).astype(np.float64)
        safe = np.where(self.mask > 0, raw_std, 1.0)
        self.std = std if std is not None else safe

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ids = r["src_ids"][:SEQ]
        mask = [1] * len(ids)
        pad = SEQ - len(ids)
        num = self.mask * (np.array(r["num33"]) - self.mean) / self.std
        return {"input_ids": torch.tensor(ids + [0] * pad, dtype=torch.long),
                "attention_mask": torch.tensor(mask + [0] * pad,
                                               dtype=torch.long),
                "numeric_features": torch.tensor(num, dtype=torch.float),
                "target": torch.tensor([r["y"]], dtype=torch.float),
                "sigma": torch.tensor([r["sigma_log"]], dtype=torch.float),
                "kernel_hash": r["hash"]}


def evaluate(model, device, rows, mean, std, batch_size=64, mask=None):
    model.eval()
    errs, biases, preds = [], [], {}
    loader = DataLoader(TTDataset(rows, mean, std, mask), batch_size=batch_size,
                        shuffle=False)
    with torch.no_grad():
        for b in loader:
            mu, vp = model(b["input_ids"].to(device),
                           b["attention_mask"].to(device),
                           b["numeric_features"].to(device))
            for h, m in zip(b["kernel_hash"],
                            mu.squeeze(-1).cpu().tolist()):
                preds[h] = m
    for r in rows:
        e = preds[r["hash"]] - r["y"]
        errs.append(abs(e))
        biases.append(e)
    return errs, biases, preds


def report(rows, errs, biases):
    out = {}
    for key in ("triple", "regime", "op"):
        groups = {}
        for r, e, b in zip(rows, errs, biases):
            groups.setdefault(r[key], []).append((e, b))
        out[key] = {g: {"n": len(v),
                        "mean_abs_err": sum(e for e, _ in v) / len(v),
                        "signed_bias": sum(b for _, b in v) / len(v)}
                    for g, v in sorted(groups.items())}
    return out


def run(data_path, out_dir, epochs, d_model, n_heads, n_layers, d_ff,
        resume=None, arm_tag="full", fusion_type="concat"):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rows = [json.loads(l) for l in Path(data_path).read_text().splitlines()]
    dig = {}
    for f in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl"):
        for line in (Path(data_path).parents[2] / f).read_text().splitlines():
            r = json.loads(line)
            dig[r["kernel_hash"]] = r["token_digest"]
    gsize = {}
    for r in rows:
        gsize[dig[r["hash"]]] = gsize.get(dig[r["hash"]], 0) + 1
    ordered = sorted(gsize, key=lambda d: (gsize[d], d))
    test_g, val_g = set(ordered[4::5]), set(ordered[3::5])
    parts = {"train": [], "val": [], "test": []}
    for r in rows:
        d = dig[r["hash"]]
        parts["test" if d in test_g else "val" if d in val_g
              else "train"].append(r)
    assert all(parts.values()) and \
        sum(map(len, parts.values())) == len(rows)
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        da = {dig[r["hash"]] for r in parts[a]}
        db = {dig[r["hash"]] for r in parts[b]}
        assert not (da & db), f"digest straddles {a}/{b}"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(vocab_size=256, max_seq_len=SEQ, d_model=d_model,
                         numeric_dim=33, use_numeric=True, dual_head=True,
                         fusion_type=fusion_type)
    n_params = sum(p.numel() for p in model.parameters())
    model.to(device)
    tr_ds = TTDataset(parts["train"])
    start_ep, step = 0, 0
    if resume:
        ck = torch.load(resume, map_location=device)
        model.load_state_dict(ck["model"])
        start_ep = ck["epoch"] + 1
        step = ck["step"]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0,
                        pin_memory=(device == "cuda"), drop_last=True)
    steps_per_epoch = max(1, len(loader))
    total = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    t0 = time.time()
    step_times = []
    for ep in range(start_ep, epochs):
        model.train()
        for b in loader:
            if step >= total:
                break
            s = time.time()
            opt.zero_grad()
            mu, vp = model(b["input_ids"].to(device),
                           b["attention_mask"].to(device),
                           b["numeric_features"].to(device))
            loss = nll_floor(mu, vp, b["target"].to(device),
                             b["sigma"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step_times.append(time.time() - s)
            step += 1
            if step % max(1, steps_per_epoch // 2) == 0:
                print(f"GPQUEUE_PROGRESS done={step} total={total} "
                      f"unit=steps", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep, "step": step},
                   out / "ckpt_last.pt")
    mean, std = tr_ds.mean.tolist(), tr_ds.std.tolist()
    rep = {"n_params": n_params, "epochs": epochs, "seed": SEED,
           "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
           "d_ff": d_ff, "lr": LR, "batch": BATCH,
           "fusion_type": fusion_type,
           "step_time_s_median_first10": sorted(step_times[:10])[5] if
           len(step_times) >= 10 else None,
           "num_norm": {"mean": mean, "std": std,
                        "mask": tr_ds.mask.tolist()},
           "split": {"test_groups": sorted(test_g), "val_groups": sorted(val_g),
                     "sizes": {k: len(v) for k, v in parts.items()}}}
    for name in ("train", "val", "test"):
        errs, biases, _ = evaluate(model, device, parts[name], tr_ds.mean,
                                   tr_ds.std, mask=tr_ds.mask)
        rep[f"{name}_report"] = report(parts[name], errs, biases)
    (out / "run_metrics.json").write_text(json.dumps(rep, indent=2))
    print(f"{arm_tag}: params={n_params} " + " ".join(
        f"{n}=" + ",".join(f"{k}:{v['mean_abs_err']:.4f}" for k, v in
        rep[f"{n}_report"]["triple"].items()) for n in ("train", "val", "test")),
        flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["scratch", "full"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-type", default="concat",
                    choices=["concat", "cross_attention"])
    args = ap.parse_args()
    if args.mode == "scratch":
        raise SystemExit("scratch mode lives in negctrl_91.py (single-family probe)")
    return 0 if run(args.data, args.out, EPOCHS, D_MODEL, N_HEADS, N_LAYERS,
                     D_FF, resume=args.resume,
                     fusion_type=args.fusion_type) else 1


if __name__ == "__main__":
    raise SystemExit(main())
