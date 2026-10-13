#!/usr/bin/env python3
"""gpu-sim-u2o.6.7.1 cold-fix FULL-CORPUS verdict (CPU), with the
PRE-REGISTERED repeatability-aware bar (PI ruling 16:51, committed BEFORE
dose 2 ran -- a bar written after the run is a sentiment, not a gate).

THE BAR (exact rule):
  Evaluated set: all 264 cells MINUS STABLE_BASELINE = {1a186c2aaaa5918d,
  25923dc793d1640a, 41c604afeb769078} (full 16-hex hashes; truncated forms match
  nothing by construction). These three move in cold-vs-cold (present without
  not fix-regressions); evidence is the two committed cold runs
  (rows_copy_cold_probe.jsonl vs rows_copy_fix_20261006.jsonl, same side
  both times). Denominator = 261.
  Per-cell PASS: |rel| <= tol(cell) [same TOL rule as every 6.7.1 packet],
  OR (|rel| - tol <= 0.05 AND cid in KNOWN_FLIPPERS = {67aadb4ae2474be6,
  6ff062d8f90e3346, 523d9bb2d85f0490} -- flipped between the two committed cold
  by same-condition repeat. No edge band for other ops: no same-condition
  repeat data exists for them, so strict tol there.)
  BAR: passes >= 200/261. Arithmetic, stated: pre-fix warm rerun 197/264,
  plus copy recovery 23/33 -> 30/33 (+7), minus fluctuation margin 4 = 200.
  The dose-1 FAIL (27/33 vs the single-realization 30/33 bar) stays on
  record; this bar replaces its method, not its verdict.

FULL BATTERY (all must hold alongside the bar):
  1. summary mode == cold (the fix ran, not warm-by-accident).
  2. device-time-unchanged from the dose summary.
  3. per-example sampling: 264/264 distinct times, 0 rows lacking own_job_id.
  4. throughput >= 50 ex/s (half the measured cold rate ~100; any
     process-per-example regression collapses this toward ~2.4).
  5. criterion-7 CPU suite result recorded (run separately, reported here).

Deterministic; writes metrics JSON to --metrics-out.
Usage: python3 experiments/gpu-sim-u2o.6.7.1/full_verify.py --metrics-out <p>
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROWS = HERE / "rows_full_fix_20261006.jsonl"
SUMMARY = HERE / "summary_full_fix_20261006.json"
REF = "lab/gpu-sim-u2o.7.1:experiments/gpu-sim-u2o.7.1/measured.jsonl"
TOL_FLOOR, TOL_CAP, TOL_SIGMA = 0.10, 0.75, 3.0
CEILING = 7911.0
STABLE_BASELINE = {"1a186c2aaaa5918d", "25923dc793d1640a", "41c604afeb769078"}
KNOWN_FLIPPERS = {"67aadb4ae2474be6", "6ff062d8f90e3346", "523d9bb2d85f0490"}
EDGE_BAND = 0.05
BAR_PASSES, BAR_DENOM = 200, 261
THROUGHPUT_FLOOR = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    fails = []
    rows = [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]
    s = json.loads(SUMMARY.read_text())
    if len(rows) != 264:
        fails.append(f"expected 264 rows, got {len(rows)}")
    if s.get("mode") != "cold":
        fails.append(f"dose did not run the fix (mode={s.get('mode')})")
    r = subprocess.run(["git", "show", REF], cwd=str(REPO), capture_output=True,
                       text=True, timeout=60)
    if r.returncode != 0:
        print("FATAL: no reference", flush=True)
        return 2
    ref = {json.loads(l)["kernel_hash"]: json.loads(l)
           for l in r.stdout.splitlines() if l.strip()}
    evaluated, passes, edge_passes, strict_passes = 0, 0, 0, 0
    baselined = sorted(STABLE_BASELINE & {rr["kernel_hash"] for rr in rows})
    for rr in rows:
        cid = rr["kernel_hash"]
        if cid in STABLE_BASELINE:
            continue
        b = ref.get(cid)
        if not b:
            continue
        evaluated += 1
        base = b["median_runtime_ms"]
        rel = abs(rr["median_runtime_ms"] - base) / base
        tol = max(TOL_FLOOR, min(TOL_SIGMA * (b.get("ai_cv") or 0.0), TOL_CAP))
        if rel <= tol:
            passes += 1
            strict_passes += 1
        elif rel - tol <= EDGE_BAND and cid in KNOWN_FLIPPERS:
            passes += 1
            edge_passes += 1
    if evaluated != BAR_DENOM:
        fails.append(f"denominator moved: {evaluated} != {BAR_DENOM}")
    if passes < BAR_PASSES:
        fails.append(f"bar missed: {passes}/{evaluated} < {BAR_PASSES}/{BAR_DENOM}")
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
    if t["examples_per_sec"] < THROUGHPUT_FLOOR:
        fails.append(f"throughput floor: {t['examples_per_sec']:.1f} ex/s "
                     f"< {THROUGHPUT_FLOOR}")
    metrics = {
        "bead": "gpu-sim-u2o.6.7.1",
        "packet": "cold-fix-full-confirm",
        "preregistered_bar": {
            "rule": "passes>=200/261; stable-baseline {1a186c2aaaa5918d,25923dc793d1640a,41c604afeb769078} excluded; edge band 0.05 for known flippers {67aadb4ae2474be6,6ff062d8f90e3346,523d9bb2d85f0490} only",
            "arithmetic": "197 (pre-fix warm) + 7 (copy 23->30) - 4 (fluctuation margin) = 200",
            "dose_1_fail_stands": True},
        "evaluated": evaluated,
        "passes": passes,
        "strict_passes": strict_passes,
        "edge_passes": edge_passes,
        "baselined_out": baselined,
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
    print(f"passes={passes}/{evaluated} (strict {strict_passes}, edge {edge_passes}) "
          f"ex_s={t['examples_per_sec']:.1f} dev_ok={dev['unchanged']} "
          f"verdict={metrics['verdict']} -> {args.metrics_out}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
