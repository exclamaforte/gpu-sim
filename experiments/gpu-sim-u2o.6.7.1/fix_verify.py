#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1 cold-fix verify (CPU): the fixed worker (default cold)
must put copy reproduce back at cold levels, with throughput, device time,
and per-example sampling intact.

Gates (verdict=pass iff ALL hold):
  1. copy reproduce vs 7.1 >= 30/33 (the cold bar from the probe).
  2. throughput stated vs the 7,911 ex/s upper-bound ceiling (informational).
  3. device-time-unchanged from the dose summary.
  4. per-example sampling 33/33 distinct times, 0 rows lacking own_job_id.
Deterministic; writes metrics JSON to --metrics-out.

Usage: python3 experiments/gpu-sim-u2o.6.7.1/fix_verify.py --metrics-out <p>
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROWS = HERE / "rows_copy_fix_20261006.jsonl"
SUMMARY = HERE / "summary_copy_fix_20261006.json"
REF = "lab/gpu-sim-u2o.7.1:experiments/gpu-sim-u2o.7.1/measured.jsonl"
TOL_FLOOR, TOL_CAP, TOL_SIGMA = 0.10, 0.75, 3.0
CEILING = 7911.0
COPY_BAR = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    fails = []
    rows = [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]
    s = json.loads(SUMMARY.read_text())
    if len(rows) != 33:
        fails.append(f"expected 33 copy rows, got {len(rows)}")
    if s.get("mode") != "cold":
        fails.append(f"dose did not run the fixed default (mode={s.get('mode')})")
    r = subprocess.run(["git", "show", REF], cwd=str(REPO), capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        print("FATAL: no reference", flush=True)
        return 2
    ref = {json.loads(l)["kernel_hash"]: json.loads(l)
           for l in r.stdout.splitlines() if l.strip()}
    rep = 0
    for rr in rows:
        b = ref.get(rr["kernel_hash"])
        if not b:
            continue
        base = b["median_runtime_ms"]
        rel = abs(rr["median_runtime_ms"] - base) / base
        tol = max(TOL_FLOOR, min(TOL_SIGMA * (b.get("ai_cv") or 0.0), TOL_CAP))
        if rel <= tol:
            rep += 1
    if rep < COPY_BAR:
        fails.append(f"copy reproduce {rep}/33 below cold bar {COPY_BAR}/33")
    dev = s["device_time_per_example"]
    if not dev["unchanged"]:
        fails.append("device time moved")
    ps = s["per_example_sampling"]
    no_id = sum(1 for rr in rows
                if not (rr.get("queue_state") or {}).get("own_job_id"))
    times = {rr.get("sampled_at_epoch") for rr in rows
             if rr.get("sampled_at_epoch")}
    if ps["distinct_sample_times"] != len(rows) or no_id or len(times) < len(rows):
        fails.append(f"sampling gate: distinct={ps['distinct_sample_times']} "
                     f"no_id={no_id} times={len(times)}/{len(rows)}")
    t = s["throughput"]
    metrics = {
        "bead": "gpu-sim-u2o.6.7.1",
        "packet": "cold-fix-verify",
        "copy_reproduce_vs_7_1": f"{rep}/33",
        "copy_bar": f"{COPY_BAR}/33",
        "throughput_ex_s": round(t["examples_per_sec"], 2),
        "ceiling_ex_s": CEILING,
        "fraction_of_ceiling": round(t["examples_per_sec"] / CEILING, 4),
        "device_time_unchanged": dev["unchanged"],
        "per_example_sampling_ok": not any("sampling" in f for f in fails),
        "verdict": "pass" if not fails else "fail",
        "failures": fails,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"copy_rep={rep}/33 dev_ok={dev['unchanged']} "
          f"sampling_ok={len(times)}/{len(rows)} verdict={metrics['verdict']} "
          f"-> {args.metrics_out}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
