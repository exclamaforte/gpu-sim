#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1 copy-locus causal verdict (CPU): warmth gradient or not.

Reads the three committed copy-probe arms + the 7.1 reference and evaluates
the PI's verdict condition: first-in-batch reproduces cold medians AND later
positions run faster with a position gradient => warmth is the cause, fix is
warm-up/discard or a position covariate (stated explicitly). Anything else =>
state what the data says instead. Deterministic; --metrics-out.

Usage: python3 experiments/gpu-sim-u2o.6.7.1/copy_verdict.py --metrics-out <p>
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REF = "lab/gpu-sim-u2o.7.1:experiments/gpu-sim-u2o.7.1/measured.jsonl"
ARMS = ("warm", "cold", "perproc")
TOL_FLOOR, TOL_CAP, TOL_SIGMA = 0.10, 0.75, 3.0


def load(name):
    p = HERE / f"rows_copy_{name}_probe.jsonl"
    return {json.loads(l)["kernel_hash"]: json.loads(l)
            for l in p.read_text().splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    r = subprocess.run(["git", "show", REF], cwd=str(REPO), capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        print("FATAL: no reference", flush=True)
        return 2
    ref = {json.loads(l)["kernel_hash"]: json.loads(l)
           for l in r.stdout.splitlines() if l.strip()}
    arms = {m: load(m) for m in ARMS}
    cids = sorted(set(arms["warm"]) & set(arms["cold"]) & set(arms["perproc"])
                  & set(ref))
    tol = {c: max(TOL_FLOOR, min(TOL_SIGMA * (ref[c].get("ai_cv") or 0.0),
                                 TOL_CAP)) for c in cids}

    def rel(m, c):
        return (arms[m][c]["median_runtime_ms"] - ref[c]["median_runtime_ms"]) \
            / ref[c]["median_runtime_ms"]

    # 1. Position gradient inside the warm arm (positions are measurement order).
    wpos = sorted(((arms["warm"][c].get("position_in_batch", -1), c) for c in cids))
    assert all(p >= 0 for p, _ in wpos), "positions missing"
    assert len({p for p, _ in wpos}) == len(wpos), "positions not distinct"
    wr = [rel("warm", c) for _, c in wpos]
    third = max(1, len(wr) // 3)
    first_med = statistics.median(wr[:third])
    last_med = statistics.median(wr[-third:])
    # Spearman position-vs-rel (negative = later is faster).
    rk = sorted(range(len(wr)), key=lambda i: wr[i])
    pos_rank = [0] * len(wr)
    for rank, i in enumerate(rk):
        pos_rank[i] = rank
    n = len(wr)
    mean = (n - 1) / 2
    cov = sum((i - mean) * (pos_rank[i] - mean) for i in range(n))
    var = sum((i - mean) ** 2 for i in range(n))
    spearman = round(cov / var if var else 0.0, 4)
    gradient = (last_med < first_med) and (spearman < -0.3)

    # 2. First-in-batch vs cold, and warm-vs-cold per cell.
    first_cid = wpos[0][1]
    first_rel_cold = (arms["warm"][first_cid]["median_runtime_ms"]
                      - arms["cold"][first_cid]["median_runtime_ms"]) \
        / arms["cold"][first_cid]["median_runtime_ms"]
    first_reproduces_cold = abs(first_rel_cold) <= tol[first_cid]
    warm_faster = sum(1 for c in cids
                      if arms["warm"][c]["median_runtime_ms"]
                      < arms["cold"][c]["median_runtime_ms"])
    wvc = sorted((arms["warm"][c]["median_runtime_ms"]
                  - arms["cold"][c]["median_runtime_ms"])
                 / arms["cold"][c]["median_runtime_ms"] for c in cids)
    wvc_med = statistics.median(wvc)

    # 3. Cold vs 7.1 (does the cold control reproduce the reference?).
    cold_rep = sum(1 for c in cids if abs(rel("cold", c)) <= tol[c])
    perproc_rep = sum(1 for c in cids if abs(rel("perproc", c)) <= tol[c])

    if first_reproduces_cold and gradient:
        verdict = ("WARMTH-CONFIRMED: first-in-batch reproduces its cold median "
                   f"({first_rel_cold:+.3f} within tol {tol[first_cid]:.2f}), "
                   f"later positions run faster (first-third {first_med:+.3f} vs "
                   f"last-third {last_med:+.3f}, spearman {spearman}). FIX: "
                   "warm-up/discard (drop position-0..k or pre-touch buffers) "
                   "OR a position covariate in the estimator -- warm-up/discard "
                   "preferred: it removes the cause instead of modelling it.")
    else:
        verdict = (f"WARMTH-REJECTED-AS-STATED: first_reproduces_cold="
                   f"{first_reproduces_cold} ({first_rel_cold:+.3f}), gradient="
                   f"{gradient} (first-third {first_med:+.3f}, last-third "
                   f"{last_med:+.3f}, spearman {spearman}). Warm cells faster "
                   f"than cold: {warm_faster}/{len(cids)}. Cold reproduces 7.1: "
                   f"{cold_rep}/{len(cids)}; perproc reproduces 7.1: "
                   f"{perproc_rep}/{len(cids)}. The copy offset is NOT a "
                   "within-batch warmth gradient -- next hypothesis must come "
                   "from where cold/perproc sit relative to 7.1.")
    metrics = {
        "bead": "gpu-sim-u2o.6.7.1",
        "packet": "copy-warmth-probe",
        "n_copy_cells": len(cids),
        "position_gradient": {
            "first_third_median_rel": round(first_med, 4),
            "last_third_median_rel": round(last_med, 4),
            "spearman_position_vs_rel": spearman,
            "gradient_present": gradient},
        "first_vs_cold": {"cid": first_cid,
                          "rel": round(first_rel_cold, 4),
                          "reproduces": first_reproduces_cold},
        "warm_faster_than_cold": f"{warm_faster}/{len(cids)}",
        "warm_vs_cold_median_rel": round(wvc_med, 4),
        "cold_reproduces_7_1": f"{cold_rep}/{len(cids)}",
        "perproc_reproduces_7_1": f"{perproc_rep}/{len(cids)}",
        "verdict": verdict,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"cells={len(cids)} gradient={gradient} first_ok={first_reproduces_cold} "
          f"warm_faster={warm_faster}/{len(cids)} cold_rep={cold_rep}/{len(cids)} "
          f"-> {args.metrics_out}", flush=True)
    print(verdict, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
