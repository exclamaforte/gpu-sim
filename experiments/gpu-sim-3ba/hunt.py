#!/usr/bin/env python3
"""gpu-sim-3ba run 2: multi-sample quiet-core hunt with immediate window use.

See pre_registration.md (consolidated, §0 records that run 1 was aborted because
its pre-registration and script were replaced mid-run by another session).

20 samples x 5s, 30s apart; threshold 0.25 unchanged from ep3.
If any sample's least-busy core is < 0.25 -> QUIET-CORE-OBSERVED and ep3's three
windows run IMMEDIATELY on that core in this same process.

Writes: samples.jsonl (append, one line per sample), windows_window_<n>.json,
        metrics.json, verdict.json
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NCPU = 32
SAMPLE_S = 5.0
SPACING_S = 30.0
NSAMPLES = 20
UTIL_THRESHOLD = 0.25
REQUIREMENT_US = 36.2


def read_proc_stat():
    res = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            toks = line.split()
            if toks[0] == "cpu":
                continue
            idx = int(toks[0][3:])
            if idx >= NCPU:
                continue
            vals = [int(x) for x in toks[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            res[idx] = (idle, sum(vals))
    return res


def sample_utils():
    before = read_proc_stat()
    time.sleep(SAMPLE_S)
    after = read_proc_stat()
    utils = {}
    for idx in sorted(before):
        ib, tb = before[idx]
        ia, ta = after[idx]
        dt = ta - tb
        utils[idx] = round(1.0 - ((ia - ib) / dt), 4) if dt > 0 else 0.0
    return utils


def run_ep3_windows(cpu):
    out = []
    for w in (1, 2, 3):
        r = subprocess.run(
            ["taskset", "-c", str(cpu), sys.executable,
             os.path.join(HERE, "measure_window.py"),
             "--cpu", str(cpu), "--window", str(w)],
            capture_output=True, text=True)
        path = os.path.join(HERE, "windows_window_%d.json" % w)
        if os.path.exists(path):
            out.append(json.load(open(path)))
        else:
            out.append({"window": w, "invalid": True,
                        "error": r.stderr[-500:]})
    return out


def decide(windows):
    clean = [w for w in windows
             if w.get("mask_valid_pinned")
             and w.get("target_cpu_util_below_25pct")
             and w.get("spread_within_requirement")]
    n = len(clean)
    median = sorted(w["spread_us"] for w in clean)[n // 2] if n else None
    admitted = n >= 2 and median is not None and median <= REQUIREMENT_US
    return ("ADMITTED" if admitted else "NO-ADMISSION-QUIET-CORE"), clean, median


def main():
    samples, hit = [], None
    for i in range(1, NSAMPLES + 1):
        utils = sample_utils()
        target = min(utils, key=lambda k: (utils[k], k))
        rec = {
            "sample": i,
            "run": 2,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sample_s": SAMPLE_S,
            "loadavg": open("/proc/loadavg").read().split()[:3],
            "least_busy_cpu": target,
            "least_busy_util": utils[target],
            "utils": utils,
        }
        samples.append(rec)
        with open(os.path.join(HERE, "samples.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("sample %2d/%d  least_busy=cpu%-2d util=%.4f  %s"
              % (i, NSAMPLES, target, utils[target], rec["timestamp_utc"]),
              flush=True)
        if utils[target] < UTIL_THRESHOLD:
            hit = rec
            print("QUIET CORE: cpu %d at %.4f -- running ep3 windows NOW"
                  % (target, utils[target]), flush=True)
            break
        if i < NSAMPLES:
            time.sleep(SPACING_S)

    minima = [s["least_busy_util"] for s in samples]
    result = {
        "bead": "gpu-sim-3ba",
        "run": 2,
        "executor": "PI (implementer wedged three times; its concurrent write to this worktree aborted run 1 -- see pre_registration.md §0)",
        "pre_registration": "experiments/gpu-sim-3ba/pre_registration.md",
        "nsamples": len(samples),
        "samples_planned": NSAMPLES,
        "spacing_s": SPACING_S,
        "sample_s": SAMPLE_S,
        "threshold": UTIL_THRESHOLD,
        "requirement_us": REQUIREMENT_US,
        "minima_of_per_sample_least_busy": minima,
        "min_minimum": min(minima),
        "median_minimum": sorted(minima)[len(minima) // 2],
        "max_minimum": max(minima),
        "n_samples_below_threshold": sum(1 for m in minima if m < UTIL_THRESHOLD),
        "quiet_core_observed": hit is not None,
        "excluded_run1_samples": "/home/gabe/aborted-3ba-run1/samples_run1.jsonl (3 samples, files replaced mid-run)",
    }
    if hit is not None:
        result["verdict"] = "QUIET-CORE-OBSERVED"
        result["hit"] = {k: hit[k] for k in
                         ("sample", "timestamp_utc", "least_busy_cpu",
                          "least_busy_util")}
        windows = run_ep3_windows(hit["least_busy_cpu"])
        v, clean, median = decide(windows)
        result["ep3_windows"] = windows
        result["ep3_windows_clean_count"] = len(clean)
        result["ep3_median_spread_us_of_clean"] = median
        result["ep3_verdict"] = v
    else:
        result["verdict"] = "NO-QUIET-CORE-IN-20-SAMPLES"

    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(HERE, "verdict.json"), "w") as f:
        json.dump({"verdict": result["verdict"],
                   "min_minimum": result["min_minimum"],
                   "n_samples_below_threshold":
                       result["n_samples_below_threshold"],
                   "ep3_verdict": result.get("ep3_verdict")}, f, indent=2)
    print(json.dumps({k: result[k] for k in
                      ("verdict", "nsamples", "min_minimum", "median_minimum",
                       "max_minimum", "n_samples_below_threshold")}, indent=2))


if __name__ == "__main__":
    main()
