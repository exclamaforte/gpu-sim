#!/usr/bin/env python3
"""gpu-sim-u2o.9.1 re-eval (CPU): recompute reports from an existing ckpt.

Justified without retraining ONLY because the constant-feature mask fix
leaves train/val model inputs bit-identical to training time (constant
features were exactly 0 = mean then, and are masked to 0 now; live
features differ by <1e-9 relative). Test inputs change -- that is the fix.
Asserts input-identity on train before rewriting reports.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))

import train_trans_91 as T  # noqa: E402


def main() -> int:
    ckpt_path, data_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = [json.loads(l) for l in Path(data_path).read_text().splitlines()]
    ck = torch.load(ckpt_path, map_location="cpu")
    model = T.create_model(vocab_size=256, max_seq_len=T.SEQ, d_model=512,
                           n_heads=8, n_layers=12, d_ff=2048, numeric_dim=33,
                           use_numeric=True, dual_head=True)
    model.load_state_dict(ck["model"])
    dig = {}
    for f in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl"):
        for line in (REPO / f).read_text().splitlines():
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
    tr_ds = T.TTDataset(parts["train"])
    # Input-identity proof: old norm vs new norm agree on train/val.
    x = np.array([r["num33"] for r in parts["train"] + parts["val"]])
    old = (x - tr_ds.mean) / (tr_ds.std + 1e-9)
    new = tr_ds.mask * (x - tr_ds.mean) / np.where(tr_ds.mask > 0,
                                                  tr_ds.std, 1.0)
    assert np.abs(old - new).max() < 1e-6, "train/val inputs changed -- retrain required"
    print(f"input-identity max diff: {np.abs(old - new).max():.2e}")
    rep = {"split": {"test_groups": sorted(test_g), "val_groups": sorted(val_g)}}
    for name in ("train", "val", "test"):
        errs, biases, _ = T.evaluate(model, "cpu", parts[name], tr_ds.mean,
                                     tr_ds.std, mask=tr_ds.mask)
        rep[f"{name}_report"] = T.report(parts[name], errs, biases)
    Path(out_path).write_text(json.dumps(rep, indent=2))
    for n in ("train", "val", "test"):
        print(n, {k: round(v["mean_abs_err"], 4) for k, v in
                  rep[f"{n}_report"]["triple"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
