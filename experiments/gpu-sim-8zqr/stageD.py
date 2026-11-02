#!/usr/bin/env python3
"""Stage D: train GBT (2al confirmed config) on 5 seen families, evaluate on unseen."""
import json, math, pathlib, sys, random
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

def aggregate_verdict(per_family, mde_exceeds_effect=False):
    """Monotone aggregation. No residual demotion.

    per_family: dict family -> {label: BETTER|TIE|WORSE, degradation_factor_gbt, paired_delta_ci, ...}
    Returns one of GENERALIZES / GENERALIZES_WITH_DEGRADATION / DOES_NOT_GENERALIZE / INCONCLUSIVE.
    """
    labels = [per_family[f]["label"] for f in per_family]
    if any(l == "WORSE" for l in labels):
        return "DOES_NOT_GENERALIZE"
    if mde_exceeds_effect:
        return "INCONCLUSIVE"
    max_deg = max(per_family[f]["degradation_factor_gbt"] for f in per_family) if per_family else 0.0
    if max_deg <= 2.0:
        return "GENERALIZES"
    return "GENERALIZES_WITH_DEGRADATION"

if __name__ == "__main__":
    # seen data
    seen_rows = [json.loads(l) for l in (REPO / "experiments/gpu-sim-2al/dataset.jsonl").read_text().splitlines()]
    # phys features
    def phys_from_row(r, family_override=None):
        fam = family_override or r["family"]
        n = r["size"]; b = r["block"]
        if fam == "matmul":
            flops = n * 64.0
        elif fam in ("gemm_naive",):
            md = r.get("metadata", {})
            M = md.get("M", 1); N = md.get("N", 1); K = md.get("K", 1)
            flops = M * N * K if (M*N*K) else n
        else:
            flops = n  # memory-access / elementwise / reduction / softmax / transpose / divergent
        bm = 2 * n * 4.0
        ai = flops / bm if bm > 0 else 0.0
        grid = min((n + b - 1) // b, 65535)
        waves = grid / 170.0
        return [math.log1p(n), b, bm, flops, ai, grid, waves]

    # For seen rows, family is in r["family"]; flops rule per family
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

    # Load unseen powered
    powered = [json.loads(l) for l in (HERE / "powered_usable.jsonl").read_text().splitlines()]
    # probe
    probe = [json.loads(l) for l in (HERE / "measured_probe.jsonl").read_text().splitlines()]
    probe_usable = [r for r in probe if r.get("target_usable")]
    for r in probe_usable:
        r["y"] = math.log(r["kernel_us"]) if r.get("kernel_us",0)>0 else None

    def make_xy(rows, phys_fn):
        X = np.array([phys_fn(r) for r in rows], dtype=np.float32)
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        return X, y

    # seen train/test split (2al style: per family sort by (size,block), hold out last 20%)
    train_seen = []; test_seen = []
    for fam in sorted(set(r["family"] for r in seen_rows)):
        fr = sorted([r for r in seen_rows if r["family"] == fam], key=lambda r: (r["size"], r["block"]))
        cut = int(0.8 * len(fr))
        train_seen.extend(fr[:cut])
        test_seen.extend(fr[cut:])

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from scipy.stats import spearmanr, kendalltau

    SEEDS = [0, 1, 2]
    def train_gbt(seed):
        Xtr, ytr = make_xy(train_seen, seen_phys)
        g = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.05,
                                          min_samples_leaf=5, random_state=seed)
        g.fit(Xtr, ytr)
        return g
    def train_ridge():
        Xtr, ytr = make_xy(train_seen, seen_phys)
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
        yte = np.array([r["y"] if r.get("y") is not None else math.log(r["kernel_us"]) for r in rows], dtype=np.float32)
        mae = float(np.mean(np.abs(pred - yte)))
        rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
        # MAPE in original space
        mape = float(np.mean(np.abs(np.exp(pred) - np.exp(yte)) / np.exp(yte))) * 100
        sp = spearmanr(pred, yte).correlation
        kt = kendalltau(pred, yte).correlation
        return {"mae": mae, "rmse": rmse, "mape": mape, "spearman": float(sp), "kendall": float(kt), "preds": pred.tolist(), "ytrue": yte.tolist()}

    # seen CV reference (GBT pooled MAE on test_seen)
    seen_gbt_maes = []
    for seed in SEEDS:
        g = train_gbt(seed)
        res = eval_model(g, test_seen, seen_phys)
        seen_gbt_maes.append(res["mae"])
    seen_gbt_mae = float(np.mean(seen_gbt_maes))
    seen_floor_res = eval_model(ridge, test_seen, seen_phys, is_ridge=True)
    seen_floor_mae = seen_floor_res["mae"]

    # unseen phys fn
    def powered_phys(r):
        return r["phys"]["n"], r["phys"]["block"], r["phys"]["bm"], r["phys"]["flops"], r["phys"]["ai"], r["phys"]["grid"], r["phys"]["waves"]
    def powered_phys_vec(r):
        p = r["phys"]
        return [math.log1p(p["n"]), p["block"], p["bm"], p["flops"], p["ai"], p["grid"], p["waves"]]

    def probe_phys_vec(r):
        p = r["phys"]
        n = p["n"]
        flops = p["flops"]
        bm = p["bm"]
        ai = flops / bm if bm > 0 else 0.0
        grid = p["grid"]
        waves = p["waves"]
        block = p["block"]
        return [math.log1p(n), block, bm, flops, ai, grid, waves]

    # Bootstrap CI helper
    def bootstrap_mae(preds, ytrue, n_boot=1000, seed=0):
        rng = random.Random(seed)
        n = len(preds)
        maes = []
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            mae = sum(abs(preds[i] - ytrue[i]) for i in idx) / n
            maes.append(mae)
        maes.sort()
        lo = maes[int(0.025 * n_boot)]
        hi = maes[int(0.975 * n_boot)]
        return lo, hi

    def bootstrap_paired(preds_gbt, preds_floor, ytrue, n_boot=1000, seed=0):
        # per-cell delta = gbt_err - floor_err ; bootstrap over cells
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

    # Evaluate powered families
    results = {"per_family": {}, "probe": {}, "seen_reference": {"gbt_mae": seen_gbt_mae, "floor_mae": seen_floor_mae}}
    powered_families = ["strided_read", "interleaved_read", "block_cyclic"]
    for fam in powered_families:
        rows_fam = [r for r in powered if r["family"] == fam and r["target_usable"]]
        # per seed
        seed_maes_gbt = []; seed_maes_floor = []
        seed_res = []
        for seed in SEEDS:
            g = train_gbt(seed)
            res_gbt = eval_model(g, rows_fam, powered_phys_vec)
            res_floor = eval_model(ridge, rows_fam, powered_phys_vec, is_ridge=True)
            seed_maes_gbt.append(res_gbt["mae"])
            seed_maes_floor.append(res_floor["mae"])
            seed_res.append((res_gbt, res_floor))
        # use seed 0 for bootstrap/preds
        g0 = train_gbt(0)
        res_gbt0 = eval_model(g0, rows_fam, powered_phys_vec)
        res_floor0 = eval_model(ridge, rows_fam, powered_phys_vec, is_ridge=True)
        lo, hi = bootstrap_mae(res_gbt0["preds"], res_gbt0["ytrue"], 1000, 0)
        paired_lo, paired_hi, paired_mean = bootstrap_paired(res_gbt0["preds"], res_floor0["preds"], res_gbt0["ytrue"], 1000, 0)
        # size decades
        decades = {}
        for r, pg, pf in zip(rows_fam, res_gbt0["preds"], res_floor0["preds"]):
            n = r["size"]
            if n < 1_000_000: dec = "262K-1M"
            elif n < 4_000_000: dec = "1M-4M"
            else: dec = "4M-16M"
            decades.setdefault(dec, []).append((pg, pf, r["y"]))
        decade_summary = {}
        for dec, vals in decades.items():
            pg = [v[0] for v in vals]; pf = [v[1] for v in vals]; yt = [v[2] for v in vals]
            decade_summary[dec] = {
                "n": len(vals),
                "gbt_mae": float(np.mean(np.abs(np.array(pg)-np.array(yt)))),
                "floor_mae": float(np.mean(np.abs(np.array(pf)-np.array(yt)))),
            }
        gbt_mae = float(np.mean(seed_maes_gbt))
        floor_mae = float(np.mean(seed_maes_floor))
        deg_gbt = gbt_mae / seen_gbt_mae
        deg_floor = floor_mae / seen_floor_mae
        results["per_family"][fam] = {
            "n_usable": len(rows_fam),
            "gbt_mae_mean_over_seeds": gbt_mae,
            "floor_mae_mean_over_seeds": floor_mae,
            "gbt_mae_seed0": res_gbt0["mae"],
            "floor_mae_seed0": res_floor0["mae"],
            "mape_gbt": res_gbt0["mape"],
            "rmse_gbt": res_gbt0["rmse"],
            "spearman_gbt": res_gbt0["spearman"],
            "kendall_gbt": res_gbt0["kendall"],
            "bootstrap_ci_gbt_mae": [lo, hi],
            "paired_delta_mean": paired_mean,
            "paired_delta_ci": [paired_lo, paired_hi],
            "label": "BETTER" if paired_hi < 0 else ("WORSE" if paired_lo > 0 else "TIE"),
            "degradation_factor_gbt": deg_gbt,
            "degradation_factor_floor": deg_floor,
            "size_decades": decade_summary,
            "mde_pooled_pct_at_achieved_n": round((math.exp(1.96*(2**0.5)*__import__("statistics").stdev([math.log(r["kernel_us"]) for r in rows_fam if r["kernel_us"]>0])/(len(rows_fam)**0.5))-1)*100, 2) if len(rows_fam)>1 else None,
        }

    # Probe arm
    probe_rows_all = probe_usable
    # gemm phys: use stored phys (already has flops). divergent: flops=n
    def probe_phys_vec2(r):
        p = r["phys"]
        n = p["n"]
        # flops: for gemm use stored; else n
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

    probe_results = {}
    for fam in ["gemm_naive", "divergent_branch"]:
        rows_fam = [r for r in probe_usable if r["family"] == fam]
        g0 = train_gbt(0)
        res_gbt = eval_model(g0, rows_fam, probe_phys_vec2)
        res_floor = eval_model(ridge, rows_fam, probe_phys_vec2, is_ridge=True)
        catastrophic = res_gbt["mae"] > 2 * res_floor["mae"]
        probe_results[fam] = {
            "n_usable": len(rows_fam),
            "gbt_mae": res_gbt["mae"],
            "floor_mae": res_floor["mae"],
            "ratio_gbt_over_floor": res_gbt["mae"] / res_floor["mae"] if res_floor["mae"] > 0 else None,
            "catastrophic": bool(catastrophic),
            "verdict": "CATASTROPHIC" if catastrophic else "INCONCLUSIVE_LOW_N",
        }
    results["probe"] = probe_results

    # Overall powered verdict
    all_generalize = all(results["per_family"][f]["gbt_mae_mean_over_seeds"] < results["per_family"][f]["floor_mae_mean_over_seeds"] for f in powered_families)
    max_deg = max(results["per_family"][f]["degradation_factor_gbt"] for f in powered_families)
    significant_loss = any(results["per_family"][f]["paired_delta_ci"][0] > 0 for f in powered_families)

    # per-family MDE (paired, log-space) at achieved n
    mde_paired = {}
    mde_exceeds_effect = False
    for f in powered_families:
        rows_fam = [r for r in powered if r["family"] == f and r["target_usable"]]
        cells = {}
        for r in rows_fam:
            cells.setdefault((r["size"], r["block"]), []).append(r)
        deltas = []
        for (sz, blk), rs in cells.items():
            # paired delta per cell: mean log gbt - mean log floor (seed0)
            pass
        # approximate paired MDE from per-row paired deltas if available; fallback to pooled
        pf = results["per_family"][f]
        ci = pf["paired_delta_ci"]
        # MDE in log space = half-width of CI-ish: use 1.96*sigma/sqrt(n_cells). Compute from bootstrap width.
        # bootstrap CI width = 1.96 * sigma_boot; MDE_log = (ci[1]-ci[0])/2 is a proxy for 1.96*sigma/sqrt(n).
        mde_log = (ci[1] - ci[0]) / 2.0
        mde_paired[f] = mde_log
        effect_log = abs(pf["paired_delta_mean"])
        if mde_log > effect_log:
            mde_exceeds_effect = True
        results["per_family"][f]["mde_paired_log_at_achieved_n"] = round(mde_log, 6)
        results["per_family"][f]["mde_exceeds_effect"] = bool(mde_log > effect_log)

    # label already stored in per_family (BETTER/TIE/WORSE)
    for f in powered_families:
        ci = results["per_family"][f]["paired_delta_ci"]
        results["per_family"][f]["label"] = "BETTER" if ci[1] < 0 else ("WORSE" if ci[0] > 0 else "TIE")

    verdict = aggregate_verdict(results["per_family"], mde_exceeds_effect=mde_exceeds_effect)

    # computed fields (no literals in verdict)
    seen_gbt_mae = results["seen_reference"]["gbt_mae"]
    seen_floor_mae = results["seen_reference"]["floor_mae"]
    in_dist_x = seen_floor_mae / seen_gbt_mae
    off_xs = [results["per_family"][f]["floor_mae_mean_over_seeds"] / results["per_family"][f]["gbt_mae_mean_over_seeds"] for f in powered_families]
    results["margin_collapse"] = {
        "in_distribution_x": round(in_dist_x, 6),
        "off_distribution_min_x": round(min(off_xs), 6),
        "off_distribution_max_x": round(max(off_xs), 6),
    }
    results["floor_identity"] = {
        "estimator": "ridge (linear) on seen holdout (like-for-like with gbt)",
        "value": round(seen_floor_mae, 6),
        "gbt_seen": round(seen_gbt_mae, 6),
        "provenance": "experiments/gpu-sim-8zqr stageD seen_reference; 2al pooled analytic floor 2.7015 quoted separately with provenance only",
    }
    results["per_family_state"] = {f: {
            "label": results["per_family"][f]["label"],
            "paired_ci": results["per_family"][f]["paired_delta_ci"],
            "gbt_mae": results["per_family"][f]["gbt_mae_mean_over_seeds"],
            "floor_mae": results["per_family"][f]["floor_mae_mean_over_seeds"],
            "degradation_factor": results["per_family"][f]["degradation_factor_gbt"],
        } for f in powered_families}
    results["aggregate_verdict"] = verdict
    results["verdict"] = verdict
    results["verdict_reason"] = "aggregate_verdict=" + verdict + "; see per_family_state, margin_collapse, floor_identity computed fields"

    # write
    # write
    import argparse as _ap
    _ap_ = _ap.ArgumentParser()
    _ap_.add_argument("--out", default=str(HERE / "stageD_results.json"))
    _args_ = _ap_.parse_args()
    out = pathlib.Path(_args_.out)
    out.write_text(json.dumps(results, indent=2) + "\n")
    # both readings
    strict = "GENERALIZES" if all(results["per_family"][f]["gbt_mae_mean_over_seeds"] < results["per_family"][f]["floor_mae_mean_over_seeds"] for f in powered_families) else "DOES_NOT_GENERALIZE"
    stat = results["verdict"]
    print("STRICT (point-estimate):", strict, "| STATISTICAL (paired CI):", stat)
    print("per-family: better/tie/worse by paired CI:")
    for f in powered_families:
        ci = results["per_family"][f]["paired_delta_ci"]
        label = "BETTER" if ci[1] < 0 else ("WORSE" if ci[0] > 0 else "TIE")
        print(f"  {f}: {label} (paired CI [{ci[0]:.4f}, {ci[1]:.4f}])")
    print(json.dumps({k: results[k] for k in ["verdict","verdict_reason"]}, indent=2))
    for f in powered_families:
        r = results["per_family"][f]
        print(f, "n=", r["n_usable"], "gbt=", round(r["gbt_mae_mean_over_seeds"],4), "floor=", round(r["floor_mae_mean_over_seeds"],4), "deg=", round(r["degradation_factor_gbt"],2), "paired_ci=", [round(x,4) for x in r["paired_delta_ci"]])
    for f in ["gemm_naive","divergent_branch"]:
        r = results["probe"][f]
        print("probe", f, r["verdict"], "ratio=", r["ratio_gbt_over_floor"])
