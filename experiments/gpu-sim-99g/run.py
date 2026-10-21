#!/usr/bin/env python3
"""gpu-sim-99g orchestrator: hunt a quiet core (corrected instrument), then run
the three windows on it immediately, then decide by the pre-registered rule.

See pre_registration.md. Run under the 8-CPU quota:
  systemd-run --user --scope --property=CPUQuota=800% -- python3 run.py
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
MAX_SAMPLES = 20
SELECT_THRESHOLD = 0.25
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
    b = read_proc_stat()
    time.sleep(SAMPLE_S)
    a = read_proc_stat()
    out = {}
    for i in sorted(b):
        ib, tb = b[i]
        ia, ta = a[i]
        dt = ta - tb
        out[i] = round(1.0 - ((ia - ib) / dt), 4) if dt > 0 else 0.0
    return out


def run_windows(cpu):
    wins = []
    for w in (1, 2, 3):
        subprocess.run(["taskset", "-c", str(cpu), sys.executable,
                        os.path.join(HERE, "measure_window.py"),
                        "--cpu", str(cpu), "--window", str(w)],
                       capture_output=True, text=True)
        p = os.path.join(HERE, "session_%d.json" % w)
        wins.append(json.load(open(p)) if os.path.exists(p)
                    else {"window": w, "invalid": True})
    return wins


def decide(wins):
    valid = [w for w in wins if w.get("mask_valid_pinned")]
    clean = [w for w in valid if w.get("other_util_below_threshold")
             and w.get("spread_within_requirement")]
    n = len(clean)
    med = sorted(w["spread_us"] for w in clean)[n // 2] if n else None
    admitted = n >= 2 and med is not None and med <= REQUIREMENT_US
    return ("ADMITTED" if admitted else "NO-ADMISSION-CORRECTED-INSTRUMENT",
            valid, clean, med)


def main():
    samples, hit = [], None
    for i in range(1, MAX_SAMPLES + 1):
        u = sample_utils()
        t = min(u, key=lambda k: (u[k], k))
        rec = {"sample": i,
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "least_busy_cpu": t, "least_busy_util": u[t], "utils": u}
        samples.append(rec)
        with open(os.path.join(HERE, "samples.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("sample %2d  least_busy=cpu%-2d util=%.4f  %s"
              % (i, t, u[t], rec["timestamp_utc"]), flush=True)
        if u[t] < SELECT_THRESHOLD:
            hit = rec
            print("QUIET CORE cpu %d at %.4f -> running windows NOW" % (t, u[t]),
                  flush=True)
            break
        if i < MAX_SAMPLES:
            time.sleep(SPACING_S)

    res = {"bead": "gpu-sim-99g",
           "executor": "PI",
           "pre_registration": "experiments/gpu-sim-99g/pre_registration.md",
           "nsamples": len(samples),
           "minima": [s["least_busy_util"] for s in samples],
           "quiet_core_observed": hit is not None,
           "three_ba_verdict_unchanged": "NO-ADMISSION-QUIET-CORE (spreads 15.2/915.4/25.3us; excluded by the self-charging utilisation instrument)"}
    if hit is None:
        res["verdict"] = "NO-QUIET-CORE-IN-20-SAMPLES"
    else:
        res["hit"] = {k: hit[k] for k in
                      ("sample", "timestamp_utc", "least_busy_cpu",
                       "least_busy_util")}
        wins = run_windows(hit["least_busy_cpu"])
        v, valid, clean, med = decide(wins)
        res.update({"verdict": v, "windows": wins,
                    "valid_windows": len(valid), "clean_windows": len(clean),
                    "median_spread_us_of_clean": med})
    json.dump(res, open(os.path.join(HERE, "metrics.json"), "w"), indent=2)
    json.dump({"verdict": res["verdict"]}, open(os.path.join(HERE, "verdict.json"), "w"))
    print(json.dumps({k: res[k] for k in ("verdict", "nsamples")
                      if k in res}, indent=2))


if __name__ == "__main__":
    main()
