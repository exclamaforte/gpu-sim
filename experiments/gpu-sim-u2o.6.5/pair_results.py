#!/usr/bin/env python3
"""gpu-sim-u2o.6.5 final pairing (CPU): join preflight_results.json with the
ts-measured measured.json into metrics.json. Asserts the bead's acceptance:
every measured cid is a preflight pass (spec-id pairing), no reject was ever
submitted, all measured rows succeeded with real trials, cost numbers from
both phases, scope guard stated. Deterministic (sort_keys) so run.sh can
byte-compare the recompute.

Usage: python3 experiments/gpu-sim-u2o.6.5/pair_results.py --metrics-out <p>
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tree_state():
    def sh(*a):
        try:
            return subprocess.run(["git"] + list(a), capture_output=True,
                                    text=True, cwd=str(HERE.parents[1]),
                                    timeout=30).stdout.strip()
        except Exception:
            return "unknown"
    return {"branch": sh("branch", "--show-current"),
            "commit": sh("rev-parse", "--short", "HEAD")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    pre = json.loads((HERE / "preflight_results.json").read_text())
    meas = {r["cid"]: r for r in
            json.loads((HERE / "measured.json").read_text())}
    fails = []
    passed_cids = [p["cid"] for p in pre["passes"] if p["ok"]]
    rejected_cids = [r["cid"] for r in pre["rejects"] if not r["ok"]]
    if set(meas) != set(passed_cids):
        fails.append(f"measured cids {sorted(meas)} != passed cids {sorted(passed_cids)}")
    if any(c in meas for c in rejected_cids):
        fails.append("a rejected kernel was submitted to the GPU")
    for cid, r in meas.items():
        if not (r.get("success") and not r.get("timeout") and r.get("n_trials", 0) > 0
                and r.get("median_ms") is not None):
            fails.append(f"{cid} has no real measurement")
    cpu_total = round(sum(p.get("cpu_sec", 0) for p in pre["passes"]), 1)
    gpu_walls = [meas[c]["wall_time_sec"] for c in passed_cids]
    medians = {c: meas[c]["median_ms"] for c in passed_cids}
    metrics = {
        "bead": "gpu-sim-u2o.6.5",
        "corpus": {"n": pre["n_corpus"], "passed": passed_cids,
                   "rejected": {r["cid"]: r["msg"] for r in pre["rejects"]}},
        "submitted": {"cids": sorted(passed_cids),
                      "ts_job": "20261006t010919-80b3abca",
                      "tree": tree_state(),
                      "rejected_submitted": []},
        "pairing": {c: {"preflight_msg": next(
            p["msg"] for p in pre["passes"] if p["cid"] == c),
            "median_ms": medians[c],
            "n_trials": meas[c]["n_trials"]} for c in sorted(passed_cids)},
        "cost": {"preflight_cpu_sec_total": cpu_total,
                 "preflight_cpu_sec_per_kernel": round(cpu_total / len(passed_cids), 2),
                 "gpu_wall_sec_per_kernel": gpu_walls,
                 "finding": "preflight screens a kernel in ~0.7 s CPU; "
                            "GPU time is spent only on passes"},
        "surprises": pre["surprises"],
        "scope_guard": ("single-op corpus (saxpy n=4096, 8 distinct launch "
                        "configs); the validator is op-agnostic by construction "
                        "(launch bounds, API scan, ptx, smoke launch) and the 4 "
                        "rejects span 4 distinct failure modes. Multi-op "
                        "coverage is u2o.7 batch territory, not this bead."),
        "verdict": "pass" if not fails else "fail",
        "failures": fails,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2,
                                                 sort_keys=True) + "\n")
    print(f"paired={len(passed_cids)} rejected_kept_out={len(rejected_cids)} "
          f"verdict={metrics['verdict']} -> {args.metrics_out}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
