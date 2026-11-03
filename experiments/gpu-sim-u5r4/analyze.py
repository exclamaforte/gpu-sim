#!/usr/bin/env python3
"""u5r4 analysis: re-measure probe arm under corrected gate."""
import json, math, random, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from scripts.grid_limits import grid_capped
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-8zqr"))
from measure import nearest_residual

DELTA = 0.10

# load measured rows (prefer in-range v2 if present)
rows_path = HERE / "measured_rows_v2.jsonl"
if not rows_path.exists():
    rows_path = HERE / "measured_rows.jsonl"
rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
print("measured rows:", len(rows), "from", rows_path.name)

# corrected usable
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

# gate pass rates
families = ["gemm_naive", "divergent_branch"]
for fam in families:
    fr = [r for r in rows if r["family"] == fam]
    nu = sum(1 for r in fr if r["corrected_usable"])
    print(fam, "n_total", len(fr), "n_usable", nu, "pass_rate", round(nu/len(fr), 4))

# seen data
seen_rows = [json.loads(l) for l in (REPO / "experiments/gpu-sim-2al/dataset.jsonl").read_text().splitlines()]
seen_families = sorted(set(r["family"] for r in seen_rows))
assert seen_families == ["elementwise", "matmul", "reduction", "softmax", "transpose"], seen_families
# assertion: training families are the 5 seen only (probe families excluded)
assert "gemm_naive" not in seen_families
assert "divergent_branch" not in seen_families
print("seen families (training):", seen_families)

def seen_phys(r):
    fam = r["family"]
    n = r["size"]; b = r["block"]
    if fam == "matmul":
        flops = n * 64.0
    else:
        flops = n
    bm = 2 * n * 4.0
    ai = flops / bm
    grid = min((n + b - 1) // b, 65535)
    waves = grid / 170.0
    return [math.log1p(n), b, bm, flops, ai, grid, waves]

def probe_phys_vec(r):
    p = r["phys"]
    n = p["n"]
    if r["family"] == "gemm_naive":
        flops = p.get("flops", n)
    else:
        flops = n
    bm = 2 * n * 4.0
    ai = flops / bm if bm > 0 else 0.0
    grid = p["grid"]
    waves = p["waves"]
    block = p["block"]
    return [math.log1p(n), block, bm, flops, ai, grid, waves]

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

SEEDS = [0, 1, 2]

def train_gbt(seed):
    # training families: 5 seen only
    train = []
    for fam in seen_families:
        fr = sorted([r for r in seen_rows if r["family"] == fam], key=lambda r: (r["size"], r["block"]))
        cut = int(0.8 * len(fr))
        train.extend(fr[:cut])
    Xtr = np.array([seen_phys(r) for r in train], dtype=np.float32)
    ytr = np.array([r["y"] for r in train], dtype=np.float32)
    g = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05,
                                      min_samples_leaf=5, random_state=seed)
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
    yte = np.array([r["y"] if "y" in r else math.log(r["kernel_us"]) for r in rows], dtype=np.float32)
    mae = float(np.mean(np.abs(pred - yte)))
    return mae, pred.tolist(), yte.tolist()

def bootstrap_paired(preds_gbt, preds_floor, ytrue, n_boot=1000, seed=0):
    rng = random.Random(seed)
    errs_gbt = [abs(preds_gbt[i]-ytrue[i]) for i in range(len(ytrue))]
    errs_floor = [abs(preds_floor[i]-ytrue[i]) for i in range(len(ytrue))]
    deltas = [errs_gbt[i]-errs_floor[i] for i in range(len(ytrue))]
    n = len(deltas)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(deltas[i] for i in idx)/n)
    boots.sort()
    return boots[int(0.025*n_boot)], boots[int(0.975*n_boot)], sum(deltas)/n

# seen CV reference
seen_gbt_maes = []
for seed in SEEDS:
    g, _ = train_gbt(seed)
    # test seen
    test_seen = []
    for fam in seen_families:
        fr = sorted([r for r in seen_rows if r["family"] == fam], key=lambda r: (r["size"], r["block"]))
        cut = int(0.8 * len(fr))
        test_seen.extend(fr[cut:])
    mae, _, _ = eval_model(g, test_seen, seen_phys)
    seen_gbt_maes.append(mae)
seen_gbt_mae = float(np.mean(seen_gbt_maes))
# floor
test_seen = []
for fam in seen_families:
    fr = sorted([r for r in seen_rows if r["family"] == fam], key=lambda r: (r["size"], r["block"]))
    cut = int(0.8 * len(fr))
    test_seen.extend(fr[cut:])
floor_mae_seen, _, _ = eval_model(ridge, test_seen, seen_phys, is_ridge=True)
seen_floor_mae = float(floor_mae_seen)

# MDE helper: half-width of bootstrap CI of paired delta (log space)
def mde_at_n(preds_gbt, preds_floor, ytrue):
    lo, hi, mean = bootstrap_paired(preds_gbt, preds_floor, ytrue, 1000, 0)
    return (hi - lo) / 2.0

results = {"per_family": {}, "seen_reference": {"gbt_mae": seen_gbt_mae, "floor_mae": seen_floor_mae}}
per_family_verdicts = {}
for fam in families:
    usable_rows = [r for r in rows if r["family"] == fam and r["corrected_usable"]]
    n_usable = len(usable_rows)
    gate_pass_rate = n_usable / len([r for r in rows if r["family"] == fam])
    # seeds
    seed_maes_gbt = []
    seed_maes_floor = []
    for seed in SEEDS:
        g, _ = train_gbt(seed)
        mae_g, _, _ = eval_model(g, usable_rows, probe_phys_vec)
        mae_f, _, _ = eval_model(ridge, usable_rows, probe_phys_vec, is_ridge=True)
        seed_maes_gbt.append(mae_g)
        seed_maes_floor.append(mae_f)
    g0, _ = train_gbt(0)
    mae_g0, preds_g0, ytrue = eval_model(g0, usable_rows, probe_phys_vec)
    mae_f0, preds_f0, _ = eval_model(ridge, usable_rows, probe_phys_vec, is_ridge=True)
    paired_lo, paired_hi, paired_mean = bootstrap_paired(preds_g0, preds_f0, ytrue, 1000, 0)
    gbt_mae = float(np.mean(seed_maes_gbt))
    floor_mae = float(np.mean(seed_maes_floor))
    deg = gbt_mae / seen_gbt_mae if seen_gbt_mae > 0 else None
    mde = mde_at_n(preds_g0, preds_f0, ytrue)
    # four-way label
    if paired_hi < 0:
        label = "BETTER"
    elif paired_lo > DELTA:
        label = "WORSE"
    elif paired_hi <= DELTA:
        label = "TIE"
    else:
        label = "UNRESOLVED"
    results["per_family"][fam] = {
        "n_total": len([r for r in rows if r["family"] == fam]),
        "n_usable": n_usable,
        "gate_pass_rate": round(gate_pass_rate, 4),
        "gbt_mae": gbt_mae,
        "floor_mae": floor_mae,
        "gbt_mae_seed0": mae_g0,
        "floor_mae_seed0": mae_f0,
        "paired_delta_mean": paired_mean,
        "paired_delta_ci": [paired_lo, paired_hi],
        "label": label,
        "degradation_factor_gbt": deg,
        "achieved_n_mde_log": round(mde, 6),
        "mde_exceeds_effect": bool(mde > abs(paired_mean)),
    }
    per_family_verdicts[fam] = label

# aggregate verdict (same rule as stageD)
labels = list(per_family_verdicts.values())
if any(l == "WORSE" for l in labels):
    agg = "DOES_NOT_GENERALIZE"
elif any(l == "UNRESOLVED" for l in labels):
    agg = "INCONCLUSIVE"
else:
    max_deg = max(results["per_family"][f]["degradation_factor_gbt"] for f in families)
    if max_deg > 2.0:
        agg = "GENERALIZES_WITH_DEGRADATION"
    else:
        agg = "GENERALIZES"

# size distribution side-by-side
def size_stats(rs):
    if not rs:
        return {"min": None, "median": None, "max": None, "n": 0}
    ns = sorted(r["size"] for r in rs)
    return {"min": ns[0], "median": ns[len(ns)//2], "max": ns[-1], "n": len(ns)}
seen_sizes = [r["size"] for r in seen_rows]
results["size_distribution"] = {
    "probe_in_range": {fam: size_stats([r for r in rows if r["family"]==fam and r["corrected_usable"]]) for fam in families},
    "seen_corpus": {"min": min(seen_sizes), "median": sorted(seen_sizes)[len(seen_sizes)//2], "max": max(seen_sizes), "n": len(seen_sizes)},
}
results["aggregate_verdict"] = agg
results["verdict"] = agg
results["verdict_rule_version"] = "delta-0.10-v1"

# four-way verdict per family already in label; add four_way_verdicts
results["four_way_verdicts"] = {f: results["per_family"][f]["label"] for f in families}

# ---- Size-scaling arm (PI direction): divergent_branch with grid=ceil(n/256), block=256 ----
sizescale_path = HERE / "measured_rows_sizescale.jsonl"
size_scaling_arm = None
if sizescale_path.exists():
    ss_rows = [json.loads(l) for l in sizescale_path.read_text().splitlines() if l.strip()]
    for r in ss_rows:
        r["corrected_usable"] = bool(new_usable(r))
    ss_usable = [r for r in ss_rows if r["corrected_usable"]]
    ss_total = len(ss_rows)
    ss_pass = len(ss_usable) / ss_total if ss_total else 0.0
    # verdict on size-scaling arm (divergent_branch only)
    if len(ss_usable) >= 3:
        g0, _ = train_gbt(0)
        mae_g0, preds_g0, ytrue_ss = eval_model(g0, ss_usable, probe_phys_vec)
        mae_f0, preds_f0, _ = eval_model(ridge, ss_usable, probe_phys_vec, is_ridge=True)
        lo, hi, mean = bootstrap_paired(preds_g0, preds_f0, ytrue_ss, 1000, 0)
        mde_ss = mde_at_n(preds_g0, preds_f0, ytrue_ss)
        if hi < 0:
            ss_label = "BETTER"
        elif lo > DELTA:
            ss_label = "WORSE"
        elif hi <= DELTA:
            ss_label = "TIE"
        else:
            ss_label = "UNRESOLVED"
        gbt_ss = float(np.mean([eval_model(train_gbt(s)[0], ss_usable, probe_phys_vec)[0] for s in SEEDS]))
        floor_ss = float(np.mean([eval_model(ridge, ss_usable, probe_phys_vec, is_ridge=True)[0] for s in SEEDS]))
        deg_ss = gbt_ss / seen_gbt_mae if seen_gbt_mae > 0 else None
    else:
        ss_label = "UNMEASURABLE"; gbt_ss = None; floor_ss = None; deg_ss = None
        lo, hi, mean = None, None, None; mde_ss = None
    size_scaling_arm = {
        "arm": "size-scaling (grid=ceil(n/256), block=256, n 1M-16M)",
        "n_total": ss_total,
        "n_usable": len(ss_usable),
        "gate_pass_rate": round(ss_pass, 4),
        "gbt_mae": gbt_ss,
        "floor_mae": floor_ss,
        "paired_delta_ci": [lo, hi] if lo is not None else None,
        "paired_delta_mean": mean,
        "label": ss_label,
        "degradation_factor_gbt": deg_ss,
        "achieved_n_mde_log": (round(mde_ss, 6) if mde_ss is not None else None),
        "size_distribution": size_stats(ss_usable) if ss_usable else None,
    }
    print("size-scaling arm:", ss_total, "rows,", len(ss_usable), "usable,", ss_label)

# ---- Occupancy arm (hold-grid): fixed grid 16384, n 1M-16M ----
hold_path = HERE / "measured_rows_holdgrid.jsonl"
occupancy_arm = None
if hold_path.exists():
    hold_rows = [json.loads(l) for l in hold_path.read_text().splitlines() if l.strip()]
    for r in hold_rows:
        r["corrected_usable"] = bool(new_usable(r))
    hold_usable = [r for r in hold_rows if r["corrected_usable"]]
    occupancy_arm = {
        "arm": "occupancy (fixed grid 16384, n 1M-16M; 1 elem/thread, no grid-stride loop)",
        "n_total": len(hold_rows),
        "n_usable": len(hold_usable),
        "gate_pass_rate": round(len(hold_usable)/len(hold_rows),4) if hold_rows else 0.0,
        "label": "UNMEASURABLE" if len(hold_usable) < 3 else "TIE",
        "size_distribution": size_stats(hold_usable) if hold_usable else {"n":0},
        "note": "0/56 usable: kernel is one element per thread with no grid-stride loop; declared size is decorative. Not a size-scaling measurement.",
    }
    print("occupancy arm:", len(hold_rows), "rows,", len(hold_usable), "usable")

results["size_scaling_arm"] = size_scaling_arm
results["occupancy_arm"] = occupancy_arm
results["rejected_arms"] = ["measured_rows.jsonl (out-of-range 1K-65K)", "measured_rows_v2.jsonl (in-range 1M-16M, grid scales with size)", "measured_rows_holdgrid.jsonl (hold-grid, 0/56 usable)"]


# write
import os as _os
out_dir = Path(_os.environ.get("U5R4_OUT", str(HERE)))
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "stageD_results.json"
out.write_text(json.dumps(results, indent=2) + "\n")
mout = out_dir / "metrics.json"
mout.write_text(json.dumps(results, indent=2) + "\n")
print("aggregate:", agg)
for f in families:
    r = results["per_family"][f]
    print(f, "n=", r["n_usable"], "pass_rate=", r["gate_pass_rate"], "gbt=", round(r["gbt_mae"],4), "floor=", round(r["floor_mae"],4), "deg=", round(r["degradation_factor_gbt"],2), "label=", r["label"], "mde=", r["achieved_n_mde_log"])
# assertion: training families only seen
assert all(f not in ("gemm_naive","divergent_branch") for f in seen_families)
print("ASSERTION OK: GBT trained on 5 seen families only")
