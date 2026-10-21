#!/usr/bin/env python3
"""gpu-sim-52i orchestrator: topology-aware quiet-PAIR hunt, then the three
windows immediately on the selected physical core.

Prediction (pre-registered, falsifiable): if the bimodality seen in 3ba/99g is
SMT sibling contention, windows on a both-siblings-quiet PHYSICAL core will be
consistently inside 36.2us. Refutation branch is in pre_registration.md §5.

Run under the 8-CPU quota:
  systemd-run --user --scope --property=CPUQuota=800% -- python3 run.py
"""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_S = 5.0
SPACING_S = 30.0
MAX_SAMPLES = 20
THRESHOLD = 0.25
REQUIREMENT_US = 36.2


def pair_map():
    pairs, seen = {}, set()
    for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/"
                       "thread_siblings_list"):
        c = int(p.split("/")[5][3:])
        txt = open(p).read().strip()
        s = tuple(sorted(int(x) for x in txt.replace("-", ",").split(",") if x))
        if c in seen:
            continue
        seen.update(s)
        pairs[c] = s
    return pairs


def read_proc_stat(cpus):
    res = {}
    for line in open("/proc/stat"):
        if not line.startswith("cpu") or line.split()[0] == "cpu":
            continue
        t = line.split()
        i = int(t[0][3:])
        if i not in cpus:
            continue
        v = [int(x) for x in t[1:]]
        res[i] = (v[3] + (v[4] if len(v) > 4 else 0), sum(v))
    return res


def sample(pairs):
    cpus = {c for s in pairs.values() for c in s}
    b = read_proc_stat(cpus)
    time.sleep(SAMPLE_S)
    a = read_proc_stat(cpus)
    u = {}
    for c in cpus:
        ib, tb = b[c]
        ia, ta = a[c]
        u[c] = round(1.0 - ((ia - ib) / (ta - tb)), 4) if ta > tb else 0.0
    return u


def run_windows(cpu):
    out = []
    for w in (1, 2, 3):
        subprocess.run(["taskset", "-c", str(cpu), sys.executable,
                        os.path.join(HERE, "measure_window.py"),
                        "--cpu", str(cpu), "--window", str(w)],
                       capture_output=True, text=True)
        p = os.path.join(HERE, "session_%d.json" % w)
        out.append(json.load(open(p)) if os.path.exists(p)
                   else {"window": w, "invalid": True})
    return out


def decide(wins):
    valid = [w for w in wins if w.get("mask_valid_pinned")]
    clean = [w for w in valid if w.get("other_util_below_threshold")
             and w.get("spread_within_requirement")]
    n = len(clean)
    med = sorted(w["spread_us"] for w in clean)[n // 2] if n else None
    admitted = n >= 2 and med is not None and med <= REQUIREMENT_US
    return ("ADMITTED" if admitted else "NO-ADMISSION-BOTH-QUIET", valid, clean, med)


def main():
    pairs = pair_map()
    samples, hit = [], None
    for i in range(1, MAX_SAMPLES + 1):
        u = sample(pairs)
        best, best_worst = None, 2.0
        for c in sorted(pairs):
            s = pairs[c]
            worst = max(u.get(x, 1.0) for x in s)
            if worst < best_worst:
                best, best_worst = c, worst
        rec = {"sample": i,
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "loadavg": open("/proc/loadavg").read().split()[:3],
               "best_pair": list(pairs[best]),
               "best_pair_utils": [u.get(x) for x in pairs[best]],
               "best_pair_worst": round(best_worst, 4),
               "utils": {str(k): v for k, v in u.items()}}
        samples.append(rec)
        with open(os.path.join(HERE, "samples.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("sample %2d  best pair cpu%s utils=%s worst=%.4f"
              % (i, pairs[best], [u.get(x) for x in pairs[best]], best_worst),
              flush=True)
        if best_worst < THRESHOLD:
            hit = rec
            print("BOTH SIBLINGS QUIET -> running windows on cpu %d NOW" % best,
                  flush=True)
            break
        if i < MAX_SAMPLES:
            time.sleep(SPACING_S)

    res = {"bead": "gpu-sim-52i",
           "executor": "PI",
           "pre_registration": "experiments/gpu-sim-52i/pre_registration.md",
           "physical_pairs": {str(k): list(v) for k, v in pairs.items()},
           "nsamples": len(samples),
           "best_pair_worst_distribution": [s["best_pair_worst"] for s in samples],
           "ninety_nine_g_verdict_unchanged": "NO-ADMISSION-CORRECTED-INSTRUMENT",
           "three_ba_verdict_unchanged": "NO-ADMISSION-QUIET-CORE"}
    if hit is None:
        res["verdict"] = "NO-BOTH-QUIET-CORE-IN-20-SAMPLES"
    else:
        res["hit"] = hit
        cpu = hit["best_pair"][0]
        wins = run_windows(cpu)
        v, valid, clean, med = decide(wins)
        res.update({"verdict": v, "pinned_cpu": cpu, "siblings": hit["best_pair"],
                    "windows": wins, "valid_windows": len(valid),
                    "clean_windows": len(clean),
                    "median_spread_us_of_clean": med})
    json.dump(res, open(os.path.join(HERE, "metrics.json"), "w"), indent=2)
    json.dump({"verdict": res["verdict"]},
              open(os.path.join(HERE, "verdict.json"), "w"))
    print(json.dumps({"verdict": res["verdict"], "nsamples": res["nsamples"]},
                     indent=2))


if __name__ == "__main__":
    main()
