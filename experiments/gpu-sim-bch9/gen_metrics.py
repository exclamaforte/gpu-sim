#!/usr/bin/env python3
"""Fabric-band robustness at 100x8: which config wins across the band."""
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import scaleout_answers as sa
from epic_cluster.parallelism_search import search

FACTORS_ELQ2 = [1/1.233, 1/1.10, 1.0, 1.10, 1.233]
LO, HI = 0.81103, 1.233
# 41 log-spaced
logpts = [LO * (HI/LO)**(i/40) for i in range(41)]
factors = sorted(set(round(f, 6) for f in (logpts + FACTORS_ELQ2)))

spec = sa.build_spec(100)
n = 800
MODELS = sa.MODELS
CONVENTIONS = sa.CONVENTIONS

def feasible_configs(model, conv_scale, f):
    hw = {"peak_flops": sa.PEAK, "mem_gb": sa.HBM_GB, "spec": spec,
          "term_scale": {"cp": conv_scale * f}, "batch": 1, "recompute": 0.0,
          "opt_state": "fp32_master", "opt_peak": "default"}
    # batch handled per call; we'll set below
    _, feas = search(n, hw, model)
    feas = [c for c in feas if c.tp <= sa.MAX_TP and c.pp <= model["n_layers"]]
    return feas

rows = []
for mname, model in MODELS.items():
    for batch in (1, 8):
        for cname, conv_scale in CONVENTIONS.items():
            key = cname if batch == 1 else f"{cname}, batch={batch}"
            # base winner at f=1.0
            feas1 = feasible_configs(model, conv_scale, 1.0)
            # fix batch in hw by recomputing with batch param (search uses hw['batch'])
            def feas_for(f, b):
                hw = {"peak_flops": sa.PEAK, "mem_gb": sa.HBM_GB, "spec": spec,
                      "term_scale": {"cp": conv_scale * f}, "batch": b, "recompute": 0.0,
                      "opt_state": "fp32_master", "opt_peak": "default"}
                _, feas = search(n, hw, model)
                feas = [c for c in feas if c.tp <= sa.MAX_TP and c.pp <= model["n_layers"]]
                return feas
            feas1 = feas_for(1.0, batch)
            best1 = feas1[0] if feas1 else None
            base_winner = (f"tp={best1.tp} pp={best1.pp} cp={best1.cp} dp={best1.dp} zero={best1.zero}" if best1 else None)
            base_margin = None
            if best1 is not None:
                alt = next((c for c in feas1 if (c.tp,c.pp,c.cp,c.dp,c.zero)!=(best1.tp,best1.pp,best1.cp,best1.dp,best1.zero)), None)
                if alt:
                    base_margin = round(100*(alt.step_sum/best1.step_sum - 1), 2)

            # feasibility/memory invariance assertion across factors
            keysets = []
            memmap = []  # list of dict key->mem_gb
            for f in factors:
                fs = feas_for(f, batch)
                ks = frozenset((c.tp,c.pp,c.cp,c.dp,c.zero) for c in fs)
                keysets.append(ks)
                memmap.append({(c.tp,c.pp,c.cp,c.dp,c.zero): c.memory_gb for c in fs})
            assert all(ks == keysets[0] for ks in keysets), "feasible key set changed across factors"
            for i in range(1, len(memmap)):
                for k, v in memmap[0].items():
                    assert abs(memmap[i][k] - v) < 1e-9, f"memory_gb changed for {k}: {memmap[i][k]} vs {v}"

            # distinct winners with factor interval
            # map factor -> winner config string
            factor_winner = {}
            for f in factors:
                fs = feas_for(f, batch)
                w = fs[0] if fs else None
                factor_winner[f] = (f"tp={w.tp} pp={w.pp} cp={w.cp} dp={w.dp} zero={w.zero}" if w else None)
            # distinct winners intervals
            distinct = []
            prev_f = factors[0]
            cur = None
            for f in factors:
                w = factor_winner[f]
                if cur is None or cur['config'] != w:
                    if cur is not None:
                        cur['factor_hi'] = prev_f
                        distinct.append(cur)
                    cur = {'config': w, 'factor_lo': f, 'factor_hi': f}
                prev_f = f
            if cur is not None:
                cur['factor_hi'] = factors[-1]
                distinct.append(cur)
            # merge consecutive same? already handled

            invariant = (len(distinct) == 1)

            # base worst-case regret: max over f of (step(base,f)/step(best_f,f) - 1)
            # need step sums per config per factor
            # compute best step per factor
            best_step_per_f = {}
            base_step_per_f = {}
            for f in factors:
                fs = feas_for(f, batch)
                bs = min((c.step_sum for c in fs), default=None)
                best_step_per_f[f] = bs
                # base config step
                base_cfg = next((c for c in fs if f"tp={c.tp} pp={c.pp} cp={c.cp} dp={c.dp} zero={c.zero}" == base_winner), None)
                base_step_per_f[f] = base_cfg.step_sum if base_cfg else None
            # base worst-case regret
            regrets = []
            for f in factors:
                if base_step_per_f[f] is not None and best_step_per_f[f] is not None and best_step_per_f[f] > 0:
                    regrets.append(base_step_per_f[f]/best_step_per_f[f] - 1)
            base_worst_regret = round(100*max(regrets), 4) if regrets else 0.0

            # minimax config: argmin_c max_f(step(c,f)/step(best_f,f)-1)
            # candidate configs = union of winners across factors
            cand_configs = set(factor_winner.values())
            # map config string -> list of step sums per factor
            config_steps = {c: [] for c in cand_configs}
            for f in factors:
                fs = feas_for(f, batch)
                stepmap = {f"tp={c.tp} pp={c.pp} cp={c.cp} dp={c.dp} zero={c.zero}": c.step_sum for c in fs}
                for c in cand_configs:
                    config_steps[c].append(stepmap.get(c))
            minimax = None
            minimax_regret = None
            for c in cand_configs:
                rgs = []
                for i, f in enumerate(factors):
                    bs = best_step_per_f[f]
                    cs = config_steps[c][i]
                    if cs is not None and bs is not None and bs > 0:
                        rgs.append(cs/bs - 1)
                if not rgs:
                    continue
                maxr = max(rgs)
                if minimax_regret is None or maxr < minimax_regret:
                    minimax_regret = maxr
                    minimax = c
            minimax_differs = (minimax != base_winner)

            rows.append({
                "model": mname,
                "convention": cname,
                "batch": batch,
                "base_winner": base_winner,
                "base_margin_pct": base_margin,
                "distinct_winners": distinct,
                "invariant": invariant,
                "base_worst_case_regret_pct": base_worst_regret,
                "minimax_config": minimax,
                "minimax_worst_case_regret_pct": round(100*minimax_regret, 4) if minimax_regret is not None else None,
                "minimax_differs_from_base": minimax_differs,
            })

# aggregates
n_flipping_5 = None  # cross-check elq2 separately; compute 46 flips
n_flipping_46 = sum(1 for r in rows if not r['invariant'])
n_invariant = sum(1 for r in rows if r['invariant'])
worst_regret_max = max((r['base_worst_case_regret_pct'] for r in rows), default=0.0)
# margin separation: max base_margin among flipped vs min among invariant
flipped_margins = [r['base_margin_pct'] for r in rows if not r['invariant'] and r['base_margin_pct'] is not None]
inv_margins = [r['base_margin_pct'] for r in rows if r['invariant'] and r['base_margin_pct'] is not None]
if flipped_margins and inv_margins:
    max_flipped = max(flipped_margins)
    min_inv = min(inv_margins)
    margin_separates = max_flipped < min_inv
    margin_threshold = round((max_flipped + min_inv)/2, 4) if margin_separates else None
else:
    margin_separates = False
    margin_threshold = None

# elq2 cross-check
elq2_metrics = json.loads((ROOT/'experiments/gpu-sim-elq2/metrics.json').read_text())
elq2_base = {}
elq2_flipped = {}
for r in elq2_metrics['rows']:
    elq2_base[(r['model'], r['convention'], r['batch'])] = r['base_winner']
    elq2_flipped[(r['model'], r['convention'], r['batch'])] = r['flipped']
mismatch = []
for r in rows:
    key = (r['model'], r['convention'], r['batch'])
    if elq2_base.get(key) != r['base_winner']:
        mismatch.append(f"base_winner mismatch {key}: elq2={elq2_base.get(key)} vs bch9={r['base_winner']}")
# flipped 5 check: recompute elq2's 5 factors
FACTORS5 = FACTORS_ELQ2
n_flip5 = 0
for r in rows:
    # recompute winners over 5 factors
    mname = r['model']; model = MODELS[mname]
    # find conv_scale
    conv_scale = None
    for cname, cs in CONVENTIONS.items():
        if cname == r['convention'].split(',')[0].strip() if ',' in r['convention'] else cname == r['convention']:
            conv_scale = cs
    # simpler: map
    # recompute
    winners = set()
    for f in FACTORS5:
        fs = None
        hw = {"peak_flops": sa.PEAK, "mem_gb": sa.HBM_GB, "spec": spec,
              "term_scale": {"cp": conv_scale * f}, "batch": r['batch'], "recompute": 0.0,
              "opt_state": "fp32_master", "opt_peak": "default"}
        _, feas = search(n, hw, model)
        feas = [c for c in feas if c.tp <= sa.MAX_TP and c.pp <= model["n_layers"]]
        if feas:
            w = feas[0]
            winners.add(f"tp={w.tp} pp={w.pp} cp={w.cp} dp={w.dp} zero={w.zero}")
    flipped5 = len(winners) > 1
    if flipped5:
        n_flip5 += 1
    if elq2_flipped.get((r['model'], r['convention'], r['batch'])) != flipped5:
        mismatch.append(f"flipped5 mismatch {r['model']},{r['convention']},b{r['batch']}: elq2={elq2_flipped.get((r['model'],r['convention'],r['batch']))} vs bch9={flipped5}")

grid_artifact = (n_flipping_46 == 0 and n_flip5 > 0)

metrics = {
    "bead": "gpu-sim-bch9",
    "band": {"lo": LO, "hi": HI, "n_factors": len(factors)},
    "grid": {"n_log_spaced": 41, "n_elq2": 5, "n_total": len(factors)},
    "rows": rows,
    "aggregates": {
        "n_rows_flipping_5": n_flip5,
        "n_rows_flipping_46": n_flipping_46,
        "n_invariant": n_invariant,
        "worst_regret_pct_max": round(worst_regret_max, 4),
        "margin_separates": margin_separates,
        "margin_threshold_pct": margin_threshold,
        "grid_artifact": grid_artifact,
        "gate": "n/a (CPU-only analysis)",
    },
    "gate": "n/a (CPU-only analysis)",
}
# cross-check elq2
metrics["elq2_cross_check"] = {"mismatches": mismatch, "n_rows_flipping_5": n_flip5}

out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (pathlib.Path(__file__).resolve().parent / "metrics.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(metrics, indent=2) + "\n")
# print cross-check
print("n_rows:", len(rows))
print("n_flipping_46:", n_flipping_46)
print("n_invariant:", n_invariant)
print("n_rows_flipping_5:", n_flip5)
print("elq2 mismatches:", len(mismatch))
for mm in mismatch:
    print("  MISMATCH:", mm)
if mismatch:
    sys.exit(1)
print("worst_regret_max:", worst_regret_max)
print("margin_separates:", margin_separates, "threshold:", margin_threshold)
