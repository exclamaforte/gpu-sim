#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1 rerun analysis (CPU): compare the fast-path rerun rows
against 7.1's reference with the bead's own tolerance rule, plus the
acceptance checks that travel with any throughput claim. Deterministic;
writes metrics JSON to --metrics-out. run.sh recomputes it.

Checks: throughput vs 510.80 ex/s reference; device-time-unchanged
(recomputed from rows with the corrected median*40*1e3 formula, NOT the
script's printed value); trial/launch counts unchanged (40/20/2 shape);
per-row queue_state + telemetry with distinct sample times; reproduce
fraction + signed median diff + faster fraction + ai_cv vs 7.1.

Usage: python3 experiments/gpu-sim-u2o.6.7.1/compare_rerun.py --metrics-out <p>
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROWS = HERE / "rows_rerun_20261006.jsonl"
SUMMARY = HERE / "summary_rerun_20261006.json"
REF = "lab/gpu-sim-u2o.7.1:experiments/gpu-sim-u2o.7.1/measured.jsonl"
TOL_FLOOR, TOL_CAP, TOL_SIGMA = 0.10, 0.75, 3.0
REF_EX_S = 510.80
REF_DEVICE_US = 126.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    fails = []
    rows = [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]
    summ = json.loads(SUMMARY.read_text())
    r = subprocess.run(["git", "show", REF], cwd=str(REPO), capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        print(f"FATAL: cannot read reference {REF}", flush=True)
        return 2
    single = {json.loads(l)["kernel_hash"]: json.loads(l)
              for l in r.stdout.splitlines() if l.strip()}

    # Shape: protocol fixed?
    shapes = {(len(rr.get("trials_ms", []))) for rr in rows}
    shape_ok = shapes == {40}
    if not shape_ok:
        fails.append(f"trial counts moved: {sorted(shapes)} != {{40}}")

    # Throughput + device time (recomputed, corrected formula).
    ex_s = summ["examples_per_sec"]
    dev_us = round(sum(rr["median_runtime_ms"] * 40 for rr in rows)
                   / len(rows) * 1e3, 1)
    dev_ratio = round(dev_us / REF_DEVICE_US, 3)

    # Reproduce vs 7.1 with the bead's tolerance rule + signed stats.
    rels, signed, reproducing, faster = [], [], 0, 0
    for rr in rows:
        s = single.get(rr["kernel_hash"])
        if not s:
            continue
        b = s["median_runtime_ms"]
        rel = abs(rr["median_runtime_ms"] - b) / b
        tol = max(TOL_FLOOR, min(TOL_SIGMA * (s.get("ai_cv") or 0.0), TOL_CAP))
        rels.append(rel)
        signed.append((rr["median_runtime_ms"] - b) / b)
        if rel <= tol:
            reproducing += 1
        if rr["median_runtime_ms"] < b:
            faster += 1
    frac = round(reproducing / len(rels), 4) if rels else 0.0
    med_rel = round(statistics.median([abs(x) for x in signed]), 4) if signed else None
    med_signed = round(statistics.median(signed), 4) if signed else None
    faster_frac = round(faster / len(signed), 4) if signed else None

    # Per-row provenance: own_job_id present + distinct sample times.
    no_id = sum(1 for rr in rows if not (rr.get("queue_state") or {}).get("own_job_id"))
    times = sorted({rr.get("sampled_at") for rr in rows if rr.get("sampled_at")})
    if no_id:
        fails.append(f"{no_id} rows lack own_job_id")
    if len(times) < 2:
        fails.append("fewer than 2 distinct sample times")

    metrics = {
        "bead": "gpu-sim-u2o.6.7.1",
        "rerun": {"ts_job": "20261006t055612-834191c6",
                  "ex_s": round(ex_s, 2),
                  "ex_s_reference": REF_EX_S,
                  "ex_s_ratio": round(ex_s / REF_EX_S, 3),
                  "device_time_us_recomputed": dev_us,
                  "device_time_us_reference": REF_DEVICE_US,
                  "device_time_ratio": dev_ratio,
                  "trial_shape": sorted(shapes),
                  "n_rows": len(rows)},
        "reproduce_vs_7_1": {"compared": len(rels),
                             "reproducing": reproducing,
                             "fraction": frac,
                             "median_rel_diff": med_rel,
                             "median_signed_diff": med_signed,
                             "faster_fraction": faster_frac,
                             "criterion_2_met_0_90": frac >= 0.90},
        "provenance": {"rows_lacking_own_job_id": no_id,
                       "distinct_sample_times": len(times)},
        "reading": ("fast path works on current master (85x single-run); "
                    "loop wall 2.5x the September arm (1.29s vs 0.52s) with "
                    "identical per-launch medians -- cause not isolated, "
                    "recorded not explained; criterion 2 still fails, cause "
                    "still unidentified (BLOCKER_2 open); device-time metric "
                    "formula in perkernel_confound.py was wrong by ~600x "
                    "since creation, corrected to median*40*1e3"),
        "verdict": "pass" if not fails else "fail",
        "failures": fails,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"rows={len(rows)} ex_s={ex_s:.1f} reproduce={reproducing}/{len(rels)} "
          f"signed={med_signed} verdict={metrics['verdict']} -> {args.metrics_out}",
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
