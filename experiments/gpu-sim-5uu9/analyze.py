#!/usr/bin/env python3
"""5uu9 analysis: config-level bootstrap, 12 distinct configs x 3 reps."""
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import json, math, random, sys, subprocess
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from scripts.grid_limits import grid_capped

# Environment fields
try:
    la = open("/proc/loadavg").read().split()[:3]
    loadavg = [float(x) for x in la]
except Exception:
    loadavg = [0.0, 0.0, 0.0]
foreign_gpu_job = False
try:
    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
    foreign_gpu_job = bool(out.strip())
except Exception:
    foreign_gpu_job = False

K = 3
reps = []
for i in (1, 2, 3):
    p = HERE / f"power_rep{i}.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    reps.append(rows)
    print(f"rep{i}: {len(rows)} rows")

# census
from collections import defaultdict
# build cells keyed by (size, block) across reps: each config must appear in all 3 reps
configs = defaultdict(list)
for ri, rows in enumerate(reps):
    for r in rows:
        configs[(r["size"], r["block"])].append((ri, r))

all_keys = set()
for rows in reps:
    for r in rows:
        all_keys.add((r["size"], r["block"]))

# duplicate configuration check (before census)
seen_sigs = []
for r in reps[0]:
    sig = (r["family"], r.get("kernel_name"), r["block"], tuple(r["grid"]), r["size"])
    seen_sigs.append(sig)
from collections import Counter
dup = [s for s, c in Counter(seen_sigs).items() if c > 1]
if dup:
    print(f"REFUSE: duplicate configuration: {len(dup)} identical specs", file=sys.stderr)
    sys.exit(1)

per_config_repcount = defaultdict(lambda: [0, 0, 0])
for ri, rows in enumerate(reps):
    for r in rows:
        per_config_repcount[(r["size"], r["block"])][ri] += 1

for key, counts in per_config_repcount.items():
    if any(c != 1 for c in counts):
        print(f"REFUSE: config {key} replicate counts {counts} != [1,1,1]", file=sys.stderr)
        sys.exit(1)
    # key set consistency
    ks = set()
    for ri, rows in enumerate(reps):
        for r in rows:
            if (r["size"], r["block"]) == key:
                ks.add(frozenset(r.keys()))
    if len(ks) != 1:
        print(f"REFUSE: config {key} has differing key sets", file=sys.stderr)
        sys.exit(1)
print(f"census OK: {len(all_keys)} configs x {K} replicates")

# corrected usable per row
def new_usable(row):
    ku = row["kernel_us"]; w = row["within_std_us"]; res = row["residual_us"]; fam = row.get("family", "")
    compute_bound = (fam in ("gemm_naive", "matmul"))
    if compute_bound and res > 0:
        thr = 5 * res
    else:
        thr = 5 * max(w, res) if (w > 0 or res > 0) else 0.0
    gc = bool(grid_capped(row["grid"]))
    return (ku >= thr) and (not gc) and (ku > 0)

for rows in reps:
    for r in rows:
        r["corrected_usable"] = bool(new_usable(r))

# aggregate to configs
configs_list = []
for key in sorted(all_keys):
    size, block = key
    cell_reps = []
    for ri, rows in enumerate(reps):
        for r in rows:
            if (r["size"], r["block"]) == key:
                cell_reps.append(r)
    usable_count = sum(1 for r in cell_reps if r["corrected_usable"])
    pass_frac = usable_count / len(cell_reps)
    usable_cfg = (usable_count >= 2)  # supermajority
    median_ku = float(np.median([r["kernel_us"] for r in cell_reps]))
    within_stds = [r["within_std_us"] for r in cell_reps]
    max_within = float(max(within_stds)) if within_stds else 0.0
    grid = cell_reps[0]["grid"]
    configs_list.append({
        "size": size,
        "block": block,
        "grid": grid,
        "n_reps": len(cell_reps),
        "pass_count": usable_count,
        "pass_frac": pass_frac,
        "usable": bool(usable_cfg),
        "median_kernel_us": median_ku,
        "max_within_std_us": max_within,
        "reps": cell_reps,
    })

usable_configs = [c for c in configs_list if c["usable"]]
n_units = len(usable_configs)
print(f"configs: {len(configs_list)}, usable: {n_units}")

# leakage asserts
seen_rows = [json.loads(l) for l in (REPO / "experiments/gpu-sim-2al/dataset.jsonl").read_text().splitlines()]
seen_families = sorted(set(r["family"] for r in seen_rows))
assert seen_families == ["elementwise", "matmul", "reduction", "softmax", "transpose"], seen_families
assert "gemm_naive" not in seen_families
assert "divergent_branch" not in seen_families
print("ASSERTION OK: GBT trained on 5 seen families only")

# phys
def seen_phys(r):
    fam = r["family"]; n = r["size"]; b = r["block"]
    if fam == "matmul": flops = n * 64.0
    else: flops = n
    bm = 2 * n * 4.0
    ai = flops / bm
    grid = min((n + b - 1) // b, 65535)
    waves = grid / 170.0
    return [math.log1p(n), b, bm, flops, ai, grid, waves]

def probe_phys(c):
    n = c["size"]; b = c["block"]; grid = c["grid"][0] if isinstance(c["grid"], list) else c["grid"]
    flops = n
    bm = 2 * n * 4.0
    ai = flops / bm
    waves = grid / 170.0
    return [math.log1p(n), b, bm, flops, ai, grid, waves]

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
    return g

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

def eval_model(model, configs, is_ridge=False):
    Xte = np.array([probe_phys(c) for c in configs], dtype=np.float32)
    if is_ridge:
        Xten = (Xte - mu_r) / sd_r
        pred = model.predict(Xten)
    else:
        pred = model.predict(Xte)
    yte = np.array([math.log(c["median_kernel_us"]) for c in configs], dtype=np.float32)
    mae = float(np.mean(np.abs(pred - yte)))
    return mae, pred.tolist(), yte.tolist()

def bootstrap_paired(preds_gbt, preds_floor, ytrue, n_boot=1000, seed=0):
    rng = random.Random(seed)
    deltas = [abs(preds_gbt[i]-ytrue[i]) - abs(preds_floor[i]-ytrue[i]) for i in range(len(ytrue))]
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(deltas[i] for i in idx) / n)
    boots.sort()
    return boots[int(0.025*n_boot)], boots[int(0.975*n_boot)], sum(deltas)/n

# Label rule
if n_units < 8:
    label = "INSUFFICIENT"
else:
    g0 = train_gbt(0)
    mae_g0, preds_g0, ytrue = eval_model(g0, usable_configs)
    mae_f0, preds_f0, _ = eval_model(ridge, usable_configs, is_ridge=True)
    paired_lo, paired_hi, paired_mean = bootstrap_paired(preds_g0, preds_f0, ytrue, 1000, 0)
    seed_maes_gbt = [eval_model(train_gbt(s), usable_configs)[0] for s in SEEDS]
    seed_maes_floor = [eval_model(ridge, usable_configs, is_ridge=True)[0] for s in SEEDS]
    gbt_mae = float(np.mean(seed_maes_gbt))
    floor_mae = float(np.mean(seed_maes_floor))
    mde = (paired_hi - paired_lo) / 2.0 if (paired_hi - paired_lo) > 0 else 0.0
    if paired_hi < 0:
        label = "BETTER"
    elif paired_lo > 0.10:
        label = "WORSE"
    elif paired_hi <= 0.10:
        label = "TIE"
    else:
        label = "UNRESOLVED"
    print("paired_delta_ci:", [round(paired_lo, 4), round(paired_hi, 4)])
    print("paired_delta_mean:", round(paired_mean, 4))
    print("gbt_mae:", round(gbt_mae, 4))
    print("floor_mae:", round(floor_mae, 4))
    print("mde:", round(mde, 4))

print("\n=== 5uu9 RESULT ===")
print("n_configs_total:", len(configs_list))
print("n_usable:", n_units)
print("label:", label)

# Per-config table
print("\nper-config table:")
for c in configs_list:
    print(f"  size={c['size']} block={c['block']} pass={c['pass_count']}/{c['n_reps']} usable={c['usable']} median_kernel_us={c['median_kernel_us']:.2f} max_within_std={c['max_within_std_us']:.4f}")

# Save metrics
results = {
    "n_configs_total": len(configs_list),
    "n_usable": n_units,
    "label": label,
    "unit_rule": "distinct (size, block) configurations with >=2/3 replicates passing the corrected gate",
    "ci_basis": "configurations (mean of 3 replicates per config), not rows",
    "per_config_table": [
        {"size": c["size"], "block": c["block"], "grid": c["grid"],
         "pass_fraction": f"{c['pass_count']}/{c['n_reps']}",
         "usable": c["usable"],
         "median_kernel_us": round(c["median_kernel_us"], 4),
         "max_within_std_us": round(c["max_within_std_us"], 4)} for c in configs_list],
    "loadavg_1_5_15": loadavg,
    "foreign_gpu_job": foreign_gpu_job,
}
if n_units >= 8:
    results["paired_delta_ci"] = [round(paired_lo, 4), round(paired_hi, 4)]
    results["paired_delta_mean"] = round(paired_mean, 4)
    results["gbt_mae"] = round(gbt_mae, 4)
    results["floor_mae"] = round(floor_mae, 4)
    results["achieved_mde"] = round(mde, 4)
    results["mde_exceeds_effect"] = bool(mde > abs(paired_mean))

out_dir = Path(_os.environ.get("U5R4_OUT", str(HERE)))
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
# Do NOT also write into the repo tree: run.sh re-executes on a CPU-only gate
# host and the gate fails when a committed artifact changes under it
# (gpu-sim-5uu9). The committed metrics.json is the artifact; the regenerated
# copy belongs in the scratch dir and is diffed against it by run.sh.
print("wrote metrics.json")
