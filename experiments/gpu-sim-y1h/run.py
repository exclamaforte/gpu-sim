#!/usr/bin/env python3
"""gpu-sim-y1h: 12 back-to-back windows pinned to a FIXED logical CPU, recording
the SMT sibling's own utilisation over the span containing each window and the
spread. Correlational test only -- NO admission claim is possible from this bead.

Run: systemd-run --user --scope --property=CPUQuota=800% -- python3 run.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us  # noqa: E402

TARGET = 7           # fixed before data (52i's quieter half of the best pair)
NWINDOWS = 12
WORKLOAD_ITERS = 100000
RUNS = 4
SPAN_S = 0.200
REQUIREMENT_US = 36.2
RHO_THRESHOLD = 0.5
LO = 0.25            # range requirement: >=3 windows below this sibling util
HI = 0.45            # and >=3 windows above this
MIN_LO = 3
MIN_HI = 3


def sibling_of(cpu):
    txt = open("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list"
               % cpu).read().strip()
    s = sorted(int(x) for x in txt.replace("-", ",").split(",") if x)
    return [x for x in s if x != cpu][0]


def tick_us():
    return 1e6 / int(subprocess.check_output(["getconf", "CLK_TCK"], text=True))


def cpu_ticks(cpu):
    for line in open("/proc/stat"):
        if line.startswith("cpu%d " % cpu):
            v = [int(x) for x in line.split()[1:]]
            return v[3] + (v[4] if len(v) > 4 else 0), sum(v)
    raise RuntimeError("cpu%d missing" % cpu)


def scope_usage():
    sp = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip() \
        .splitlines()[0].split("::")[-1]
    for line in open(sp + "/cpu.stat"):
        k, v = line.split()
        if k == "usage_usec":
            return int(v)
    return 0


def workload():
    s = 0
    for i in range(WORKLOAD_ITERS):
        s += i * i
    return s


def measure_window(w):
    sib = SIB
    tu = tick_us()
    u0 = scope_usage()
    i0, t0s = cpu_ticks(sib)
    t0 = time.time()
    durs = []
    for _ in range(RUNS):
        a = time.perf_counter()
        workload()
        durs.append((time.perf_counter() - a) * 1e6)
    t1 = time.time()
    time.sleep(max(0.0, SPAN_S - (t1 - t0)))
    i1, t1s = cpu_ticks(sib)
    u1 = scope_usage()
    span_us = (time.time() - t0) * 1e6
    sib_busy_us = ((t1s - t0s) - (i1 - i0)) * tu
    mask = sorted(os.sched_getaffinity(0))
    return {
        "window": w,
        "pinned_cpu": TARGET,
        "sibling": sib,
        "mask_valid_pinned": mask == [TARGET],
        "span_us": round(span_us, 1),
        "sibling_busy_us": round(sib_busy_us, 1),
        "sibling_util": round(sib_busy_us / span_us, 4) if span_us else None,
        "our_scope_usage_us": u1 - u0,
        "run_durations_us": [round(d, 1) for d in durs],
        "spread_us": round(spread_us(durs), 1),
        "spread_within_requirement": spread_us(durs) <= REQUIREMENT_US,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "loadavg": open("/proc/loadavg").read().split()[:3],
    }


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else None


SIB = sibling_of(TARGET)


def main():
    wins = []
    for w in range(1, NWINDOWS + 1):
        rec = measure_window(w)
        wins.append(rec)
        with open(os.path.join(HERE, "windows.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("window %2d  sibling=cpu%d util=%.4f  spread=%.1f  %s"
              % (w, rec["sibling"], rec["sibling_util"], rec["spread_us"],
                 rec["timestamp_utc"]), flush=True)
    valid = [w for w in wins if w["mask_valid_pinned"]]
    xs = [w["sibling_util"] for w in valid]
    ys = [w["spread_us"] for w in valid]
    n_lo = sum(1 for v in xs if v < LO)
    n_hi = sum(1 for v in xs if v > HI)
    rho = spearman(xs, ys) if len(valid) >= 4 else None

    if len(valid) < NWINDOWS:
        verdict = "PASS-INVALID"
    elif n_lo < MIN_LO or n_hi < MIN_HI:
        verdict = "RANGE-INSUFFICIENT"
    elif rho is not None and rho >= RHO_THRESHOLD:
        verdict = "ASSOCIATION-SUPPORTED"
    else:
        verdict = "ASSOCIATION-NOT-SUPPORTED"

    res = {
        "bead": "gpu-sim-y1h",
        "executor": "PI",
        "pre_registration": "experiments/gpu-sim-y1h/pre_registration.md",
        "target_cpu": TARGET, "sibling_cpu": SIB,
        "n_windows": len(wins), "n_valid": len(valid),
        "sibling_util_range": [min(xs), max(xs)] if xs else None,
        "n_windows_sibling_below_0_25": n_lo,
        "n_windows_sibling_above_0_45": n_hi,
        "spearman_rho": round(rho, 4) if rho is not None else None,
        "rho_threshold": RHO_THRESHOLD,
        "spread_distribution": {
            "min": min(ys) if ys else None,
            "median": sorted(ys)[len(ys) // 2] if ys else None,
            "max": max(ys) if ys else None,
            "n_within_36_2us": sum(1 for y in ys if y <= REQUIREMENT_US),
        },
        "verdict": verdict,
        "invalid_windows": len(wins) - len(valid),
        "invalid_reason": ("windows whose affinity mask was not exactly the fixed "
                           "target CPU are INVALID per pre-registration and are "
                           "excluded; if all are invalid the pass yields no verdict") if len(valid) < len(wins) else None,
        "no_admission_claimed": ("correlational test only; admission requires the "
                                "conjunctive quiet-pair rule from 99g/52i, which "
                                "this bead does not attempt"),
        "prior_verdicts_unchanged": {
            "52i": "NO-BOTH-QUIET-CORE-IN-20-SAMPLES",
            "99g": "NO-ADMISSION-CORRECTED-INSTRUMENT",
            "3ba": "NO-ADMISSION-QUIET-CORE",
        },
    }
    json.dump(res, open(os.path.join(HERE, "metrics.json"), "w"), indent=2)
    json.dump({"verdict": verdict, "rho": res["spearman_rho"]},
              open(os.path.join(HERE, "verdict.json"), "w"))
    print(json.dumps({k: res[k] for k in ("verdict", "spearman_rho",
                                         "sibling_util_range",
                                         "n_windows_sibling_below_0_25",
                                         "n_windows_sibling_above_0_45")},
                     indent=2))


if __name__ == "__main__":
    main()
