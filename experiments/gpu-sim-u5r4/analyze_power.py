#!/usr/bin/env python3
"""nlgf power analysis: 3 replicates per cell, CI over cells."""
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import json, math, random, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from scripts.grid_limits import grid_capped
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-8zqr"))

DELTA = 0.10

power_path = HERE / "measured_rows_sizescale_power.jsonl"
rows = [json.loads(l) for l in power_path.read_text().splitlines() if l.strip()]
print("power rows:", len(rows))

def new_usable(row):
    kernel_us = row["kernel_us"]
    within = row["within_std_us"]
    residual = row["residual_us"]
    family = row.get("family", "")
    compute_bound = (family in ("gemm_naive", "matmul"))
    if compute_bound and residual > 0:
        threshold = 5 * residual
    else:
        threshold = 5 * max(within, residual) if (within > 0 or residual > 0) else 0.0
    gc = bool(grid_capped(row["grid"]))
    return (kernel_us >= threshold) and (not gc) and (kernel_us > 0)

for r in rows:
    r["corrected_usable"] = bool(new_usable(r))

# Aggregate to cells: mean of 3 replicates per (size, build_idx)
from collections import defaultdict
cells = defaultdict(list)
for r in rows:
    cells[(r["size"], r["build_idx"])].append(r)

cell_rows = []
for (size, bidx), reps in cells.items():
    # take first rep as template, average numeric fields
    template = dict(reps[0])
    template["kernel_us"] = float(np.mean([r["kernel_us"] for r in reps]))
    template["within_std_us"] = float(np.mean([r["within_std_us"] for r in reps]))
    template["dispatch_us"] = float(np.mean([r["dispatch_us"] for r in reps]))
    template["_n_replicates"] = len(reps)
    template["corrected_usable"] = bool(new_usable(template))
    cell_rows.append(template)

print("cells:", len(cell_rows), "replicates per cell:", cell_rows[0]["_n_replicates"])
usable = [r for r in cell_rows if r["corrected_usable"]]
print("usable cells:", len(usable), "pass rate:", round(len(usable)/len(cell_rows),4))

# seen data for training
seen_rows = [json.loads(l) for l in (REPO / "experiments/gpu-sim-2al/dataset.jsonl").read_text().splitlines()]
seen_families = sorted(set(r["family"] for r in seen_rows))

def seen_phys(r):
    fam = r["family"]; n = r["size"]; b = r["block"]
    if fam == "matmul": flops = n * 64.0
    else: flops = n
    bm = 2 * n * 4.0
    ai = flops / bm
    grid = min((n + b - 1) // b, 65535)
    waves = grid / 170.0
    return [math.log1p(n), b, bm, flops, ai, grid, waves]

def probe_phys_vec(r):
    p = r["phys"]; n = p["n"]
    if r["family"] == "gemm_naive": flops = p.get("flops", n)
    else: flops = n
    bm = 2 * n * 4.0
    ai = flops / bm if bm > 0 else 0.0
    grid = p["grid"]; waves = p["waves"]; block = p["block"]
    return [math.log1p(n), block, bm, flops, ai, grid, waves]

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

SEEDS = [0, 1, 2]
def train_gbt(seed):
    train = []
    for fam in seen_families:
        fr = sorted([r for r in seen_rows if r["family"] == fam], key=lambda r: (r["size"], r["block"]))
        cut = int(0.8 * len(fr))
        train.extend(fr[:cut])
    Xtr = np.array([seen_phys(r) for r in train], dtype=np.float32)
    ytr = np.array([r["y"] for r in train], dtype=np.float32)
    g = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05, min_samples_leaf=5, random_state=seed)
    g.fit(Xtr, ytr)
    return g, train

def train_ridge():
    train = []
    for fam in seen_families:
        fr = sorted([r for r in seen_rows if r["family"] == fam], key=lambda r: (r["size"], r["block"]))
        cut = int(0.8 * len(fr))
        train.extend(fr[:cut])
    Xtr = np.array([seen_phys(r) for r in train], dtype=np.float32)
    ytr = np.array([r["y"] for r in train], dtype=np.float32)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1
    Xtrn = (Xtr - mu) / sd
    rr = Ridge(); rr.fit(Xtrn, ytr)
    return rr, mu, sd

ridge, mu_r, sd_r = train_ridge()

def eval_model(model, rows, phys_fn, is_ridge=False):
    Xte = np.array([phys_fn(r) for r in rows], dtype=np.float32)
    if is_ridge:
        Xten = (Xte - mu_r) / sd_r
        pred = model.predict(Xten)
    else:
        pred = model.predict(Xte)
    yte = np.array([math.log(r["kernel_us"]) for r in rows], dtype=np.float32)
    mae = float(np.mean(np.abs(pred - yte)))
    return mae, pred.tolist(), yte.tolist()

def bootstrap_paired(preds_gbt, preds_floor, ytrue, n_boot=1000, seed=0):
    rng = random.Random(seed)
    deltas = [abs(preds_gbt[i]-ytrue[i]) - abs(preds_floor[i]-ytrue[i]) for i in range(len(ytrue))]
    n = len(deltas)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(deltas[i] for i in idx)/n)
    boots.sort()
    return boots[int(0.025*n_boot)], boots[int(0.975*n_boot)], sum(deltas)/n

g0, _ = train_gbt(0)
mae_g0, preds_g0, ytrue = eval_model(g0, usable, probe_phys_vec)
mae_f0, preds_f0, _ = eval_model(ridge, usable, probe_phys_vec, is_ridge=True)
paired_lo, paired_hi, paired_mean = bootstrap_paired(preds_g0, preds_f0, ytrue, 1000, 0)

seed_maes_gbt = [eval_model(train_gbt(s)[0], usable, probe_phys_vec)[0] for s in SEEDS]
seed_maes_floor = [eval_model(ridge, usable, probe_phys_vec, is_ridge=True)[0] for s in SEEDS]
gbt_mae = float(np.mean(seed_maes_gbt))
floor_mae = float(np.mean(seed_maes_floor))

def mde_at_n(preds_gbt, preds_floor, ytrue):
    lo, hi, mean = bootstrap_paired(preds_gbt, preds_floor, ytrue, 1000, 0)
    return (hi - lo) / 2.0
mde = mde_at_n(preds_g0, preds_f0, ytrue)

if len(usable) < 3:
    label = "INSUFFICIENT"
elif paired_hi < 0:
    label = "BETTER"
elif paired_lo > DELTA:
    label = "WORSE"
elif paired_hi <= DELTA:
    label = "TIE"
else:
    label = "UNRESOLVED"

print("\n=== POWER TEST RESULT ===")
print("n_cells (total):", len(cell_rows))
print("n_usable:", len(usable))
print("gate_pass_rate:", round(len(usable)/len(cell_rows),4))
print("gbt_mae:", round(gbt_mae,4))
print("floor_mae:", round(floor_mae,4))
print("paired_delta_mean:", round(paired_mean,4))
print("paired_delta_ci:", [round(paired_lo,4), round(paired_hi,4)])
print("label:", label)
print("achieved_mde:", round(mde,4))
print("mde_exceeds_effect:", bool(mde > abs(paired_mean)))

# CI over cells note
print("\nReplicate handling: 3 replicates per (size,build_idx) cell; cell mean computed per cell;")
print("CI computed over", len(usable), "usable cells (not over", len(rows), "rows).")

# Save metrics
results = {
    "n_cells_total": len(cell_rows),
    "n_usable": len(usable),
    "gate_pass_rate": round(len(usable)/len(cell_rows),4),
    "n_replicates_per_cell": 3,
    "ci_basis": "cells (mean of 3 replicates per cell), not rows",
    "gbt_mae": gbt_mae,
    "floor_mae": floor_mae,
    "paired_delta_mean": paired_mean,
    "paired_delta_ci": [paired_lo, paired_hi],
    "label": label,
    "achieved_mde": mde,
    "mde_exceeds_effect": bool(mde > abs(paired_mean)),
    "sizes": sorted(set(r["size"] for r in cell_rows)),
}
import os as _os2
_u5r4_out = _os2.environ.get("U5R4_OUT")
if not _u5r4_out:
    print("ERROR: U5R4_OUT is required")
    raise SystemExit(2)
out_dir = Path(_u5r4_out)
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "metrics_power.json"
out.write_text(json.dumps(results, indent=2) + "\n")
print("wrote", out)
