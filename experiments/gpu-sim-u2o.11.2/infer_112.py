#!/usr/bin/env python3
"""gpu-sim-u2o.11.2: per-row test predictions from 9.1's COMMITTED-shape
39M ckpt (CPU inference, no retraining) + worst-row failure analysis.

Loads runs/gpu-sim-u2o.9.1/ckpt_last.pt (present on disk, uncommitted --
weights are not the artifact; predictions.json IS and is committed).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))


def _main_tree():
    """Main-tree path without a machine-specific hard path (runs/ is
    uncommitted working state; worktrees do not share it)."""
    import subprocess
    out = subprocess.run(["git", "worktree", "list", "--porcelain"],
                         capture_output=True, text=True, cwd=REPO)
    for line in out.stdout.splitlines():
        if line.startswith("worktree "):
            return Path(line.split(" ", 1)[1])
    raise RuntimeError("main tree not found")

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_trans_91 as T  # noqa: E402
from epic_u2o.model import create_model  # noqa: E402

EXP = REPO / "experiments" / "gpu-sim-u2o.11.2"


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rows = [json.loads(l) for l in
            (REPO / "experiments/gpu-sim-u2o.9.1"
             / "train_data_91.jsonl").read_text().splitlines()]
    parts, _, _ = T.family_holdout_split(rows, REPO)
    tr_ds = T.TTDataset(parts["train"])
    te_ds = T.TTDataset(parts["test"], tr_ds.mean, tr_ds.std, tr_ds.mask)
    te_rows = parts["test"]

    model = create_model(vocab_size=256, max_seq_len=T.SEQ, d_model=512,
                         n_heads=8, n_layers=12, d_ff=2048, numeric_dim=33,
                         use_numeric=True, dual_head=True,
                         fusion_type="concat")
    ck = torch.load(_main_tree() / "runs/gpu-sim-u2o.9.1/ckpt_last.pt",
                    map_location="cpu")
    model.load_state_dict(ck["model"])
    model.eval()

    loader = DataLoader(te_ds, batch_size=16, num_workers=0)
    mus, vps = [], []
    with torch.no_grad():
        for b in loader:
            mu, vp = model(b["input_ids"], b["attention_mask"],
                           b["numeric_features"])
            mus.extend(mu.squeeze(-1).tolist())
            vps.extend(vp.squeeze(-1).tolist())
    assert len(mus) == len(te_rows) == 57
    preds = [{"hash": r["hash"], "op": r["op"], "triple": r["triple"],
              "regime": r["regime"], "y": r["y"],
              "sigma_log": r["sigma_log"], "mu": mu, "var": vp}
             for r, mu, vp in zip(te_rows, mus, vps)]
    for p in preds:
        p["abs_err"] = abs(p["y"] - p["mu"])
    (out / "predictions.json").write_text(json.dumps(preds, indent=1))

    worst = sorted(preds, key=lambda p: -p["abs_err"])[:10]
    by_regime, by_triple = {}, {}
    for p in preds:
        by_regime.setdefault(p["regime"], []).append(p["abs_err"])
        by_triple.setdefault(p["triple"], []).append(p["abs_err"])
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    analysis = {
        "n": len(preds),
        "worst10": [{"hash": p["hash"][:8], "op": p["op"],
                     "triple": p["triple"], "regime": p["regime"],
                     "y": round(p["y"], 3), "mu": round(p["mu"], 3),
                     "abs_err": round(p["abs_err"], 3),
                     "sigma_log": round(p["sigma_log"], 3)}
                    for p in worst],
        "mean_abs_by_regime": {k: round(mean(v), 4)
                               for k, v in by_regime.items()},
        "mean_abs_by_triple": {k: round(mean(v), 4)
                               for k, v in by_triple.items()},
        "worst10_regimes": {k: sum(1 for p in worst if p["regime"] == k)
                            for k in by_regime},
        "err_vs_sigma_corr": _pearson([p["abs_err"] for p in preds],
                                      [p["sigma_log"] for p in preds]),
    }
    (out / "failure_analysis.json").write_text(json.dumps(analysis, indent=2))
    print(json.dumps({"worst10_mean_abs":
                      round(mean([p["abs_err"] for p in worst]), 4),
                      "by_regime": analysis["mean_abs_by_regime"],
                      "err_sigma_corr":
                      round(analysis["err_vs_sigma_corr"], 3)}, indent=1))
    return 0


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / (va * vb) ** 0.5 if va and vb else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
