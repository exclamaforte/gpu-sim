#!/usr/bin/env python3
"""gpu-sim-u2o.9.5 STAGE 1+2 (CPU): GBT baseline over hand-engineered features.

Stage 1: hash-verified join (8.3 features x 6.6 targets, 456/456), stated
log transform + inverse with round-trip test, model_bias exclusion asserted,
evaluation harness in 9.4's units (log-space mean abs err + signed bias per
regime triple), in-sample fit labelled IN-SAMPLE with regime census.

Stage 2 (7.7 CLOSED/merged: family_holdout_v1 exists): digest-group holdout
on the 456 stage-1 rows by the stated every-5th rule, guards raising, 9.4's
committed per-regime figures LOADED (never recomputed) as the comparison
frame.

Model note: the design names XGBoost/LightGBM; neither is installed in this
environment, so sklearn HistGradientBoostingRegressor is used -- the same
model class (histogram gradient-boosted trees). Hparams below are the
implementer's, recorded.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

HGBR = {"max_iter": 300, "max_depth": 6, "learning_rate": 0.05,
        "min_samples_leaf": 5, "l2_regularization": 0.0, "random_state": 0}


def log_transform(ms: float) -> float:
    return math.log(ms + 1e-6)


def inv_transform(log_ms: float) -> float:
    return math.exp(log_ms)


def load_join():
    feats = [json.loads(l) for l in
             (REPO / "experiments/gpu-sim-u2o.8.3/features.jsonl")
             .read_text().splitlines()]
    targs = {json.loads(l)["kernel_hash"]: json.loads(l) for l in
             (REPO / "experiments/gpu-sim-u2o.6.6/targets.jsonl")
             .read_text().splitlines()}
    assert len(feats) == 456 and len(targs) == 456
    rows, shortfall = [], []
    for f in feats:
        t = targs.get(f["kernel_hash"])
        if t is None:
            shortfall.append(f["kernel_hash"])
            continue
        assert abs(f["measured_median_ms"] - t["median_runtime_ms"]) < 1e-12, \
            f"median disagree for {f['kernel_hash']}"
        assert "bias" not in " ".join(map(str.lower, f.keys())), \
            "bias field in feature row"
        assert len(f["vector"]) == 33 and f["vector_dim"] == 33
        rows.append({"hash": f["kernel_hash"], "x": f["vector"],
                     "op": f["op_type"], "y": log_transform(t["median_runtime_ms"]),
                     "ms": t["median_runtime_ms"],
                     "triple": t["roofline_triple"],
                     "regime": t["measured_regime"],
                     "model_bias_beside": t["model_bias"]})
    assert not shortfall, f"join shortfall: {shortfall}"
    return rows


def report(rows, pred_log):
    out = {}
    for key in ("triple", "regime", "op"):
        groups = {}
        for r, p in zip(rows, pred_log):
            groups.setdefault(r[key], []).append((r["y"], p))
        out[key] = {g: {"n": len(v),
                        "mean_abs_err": sum(abs(a - b) for a, b in v) / len(v),
                        "signed_bias": sum(b - a for a, b in v) / len(v)}
                    for g, v in sorted(groups.items())}
    return out


def imbalance_flag(train_rows, test_rows):
    """Census divergence detector. Returns a flag string, or '' if balanced."""
    def dist(rs):
        c = Counter(r["triple"] for r in rs)
        n = len(rs)
        return {k: c[k] / n for k in sorted(set(c) | set())} if n else {}
    dtr, dte = dist(train_rows), dist(test_rows)
    div = sum(abs(dtr.get(k, 0) - dte.get(k, 0)) for k in set(dtr) | set(dte)) / 2
    if div > 0.25:
        return (f"REGIME-IMBALANCED: train/test triple total-variation "
                f"{div:.2f} > 0.25 (train {dict(Counter(r['triple'] for r in train_rows))} "
                f"vs test {dict(Counter(r['triple'] for r in test_rows))})")
    return ""


def fit_predict(train, test):
    import numpy as np
    Xtr = np.array([r["x"] for r in train])
    ytr = np.array([r["y"] for r in train])
    Xte = np.array([r["x"] for r in test])
    m = HistGradientBoostingRegressor(**HGBR)
    m.fit(Xtr, ytr)
    return [float(v) for v in m.predict(Xte)]


def main() -> int:
    rows = load_join()
    # Transform round-trip on a known value (acceptance 2). The +1e-6 guard
    # is a KNOWN asymmetry bounded by the guard itself, not a silent one.
    known = 0.002536
    assert abs(inv_transform(log_transform(known)) - known) < 2e-6
    # 9.4 frame: LOADED, never recomputed (acceptance 6).
    r94 = json.loads((REPO / "experiments/gpu-sim-u2o.9.4/metrics.json").read_text())
    frame = {"per_regime_achieved": r94["per_regime_achieved"],
             "residual_A": r94["residual_A"], "residual_B": r94["residual_B"],
             "families": r94["families"],
             "bandwidth_convention": r94["bandwidth_convention"]}
    metrics = {"bead": "gpu-sim-u2o.9.5", "stage": "1+2 (7.7 closed)",
               "model": "sklearn.HistGradientBoostingRegressor",
               "hparams": HGBR,
               "join": {"n": len(rows), "shortfall": [],
                        "median_agreement": "features.measured_median_ms == targets.median_runtime_ms"},
               "transform": "y=ln(ms+1e-6), pred_ms=exp(pred); round-trip bounded by guard (2e-6) on 0.002536",
               "model_bias": "6.6 model_bias carried BESIDE rows for audit, never a feature (asserted: no bias key in feature rows)",
               "convention": frame["bandwidth_convention"],
               "roofline_94_frame": frame}

    # Stage 1: in-sample fit, labelled as such (acceptance 5).
    p_in = fit_predict(rows, rows)
    metrics["stage1_insample"] = {
        "label": "IN-SAMPLE: training error only; no generalisation claim",
        "regime_census": dict(Counter(r["regime"] for r in rows)),
        "triple_census": dict(Counter(r["triple"] for r in rows)),
        "report": report(rows, p_in)}

    # Stage 2: family_holdout_v1 digest rule on the 456 stage-1 rows.
    by_hash = {}
    for f in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl"):
        for line in (REPO / f).read_text().splitlines():
            r = json.loads(line)
            by_hash[r["kernel_hash"]] = r["token_digest"]
    assert set(by_hash) == {r["hash"] for r in rows}, "digest coverage != join"
    gsize = Counter(by_hash[r["hash"]] for r in rows)
    ordered = sorted(gsize, key=lambda d: (gsize[d], d))
    test_g, val_g = set(ordered[4::5]), set(ordered[3::5])
    parts = {"train": [], "val": [], "test": []}
    for r in rows:
        d = by_hash[r["hash"]]
        parts["test" if d in test_g else "val" if d in val_g
              else "train"].append(r)
    assert all(parts.values()), "empty side"
    assert sum(map(len, parts.values())) == len(rows), "rows not conserved"
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        da = {by_hash[r["hash"]] for r in parts[a]}
        db = {by_hash[r["hash"]] for r in parts[b]}
        assert not (da & db), f"digest straddles {a}/{b}"
    for s in ("val", "test"):
        assert len(parts[s]) >= 5, f"{s} under 5 rows"
    p_tr = fit_predict(parts["train"], parts["train"])
    p_te = fit_predict(parts["train"], parts["test"])
    p_va = fit_predict(parts["train"], parts["val"])
    metrics["stage2_holdout"] = {
        "name": "family_holdout_v1 (7.7 rule, stage-1 subset)",
        "rule": "digest groups sorted by (n_rows, digest); every 5th to test, offset 1 to val",
        "test_groups": sorted(test_g), "val_groups": sorted(val_g),
        "sizes": {k: {"n": len(v),
                      "measurable": sum(1 for r in v if r["regime"] == "measurable")}
                  for k, v in parts.items()},
        "shared_digest_groups_across_sides": 0,
        "train_report": report(parts["train"], p_tr),
        "val_report": report(parts["val"], p_va),
        "test_report": report(parts["test"], p_te)}

    # Negative control: fit on launch_bound only; census must flag it.
    lb = [r for r in rows if r["regime"] == "launch_bound"]
    me = [r for r in rows if r["regime"] == "measurable"]
    flag = imbalance_flag(lb, me)
    assert flag, "census failed to flag single-regime fit"
    p_neg = fit_predict(lb, me)
    metrics["negative_control"] = {
        "setup": "fit on launch_bound (283), eval on measurable (173)",
        "census_flag": flag,
        "report": report(me, p_neg)}

    (HERE / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"join={len(rows)}/456 shortfall=0")
    print("stage1 insample triple err:",
          {k: round(v["mean_abs_err"], 4) for k, v in
           metrics["stage1_insample"]["report"]["triple"].items()})
    print("stage2 test triple err:",
          {k: round(v["mean_abs_err"], 4) for k, v in
           metrics["stage2_holdout"]["test_report"]["triple"].items()})
    print("94 frame latency->latency:",
          frame["per_regime_achieved"]["latency->latency"]["median_achieved_GBps"],
          "GB/s median")
    print("negative control flag:", flag[:80])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
