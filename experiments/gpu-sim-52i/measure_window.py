#!/usr/bin/env python3
"""gpu-sim-52i: measure ONE window pinned to a logical CPU, with the instrument
extended to the PHYSICAL core (both SMT siblings).

Corrected instrument, pair-aware:
  other_cpu_us = sum_over_siblings[(total_ticks_delta - idle_ticks_delta) * tick_us]
                 - our_scope_cpu_us
  other_utilisation = other_cpu_us / span_us
Our own CPU time is subtracted ONCE (it can only run on the pinned sibling).

Writes session_<n>.json. Must be invoked under `taskset -c <pinned>`.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us  # noqa: E402

WORKLOAD_ITERS = 100000
RUNS = 4
SPAN_S = 0.200
UTIL_THRESHOLD = 0.25
REQUIREMENT_US = 36.2


def siblings(cpu):
    txt = open("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list"
               % cpu).read().strip()
    return sorted(int(x) for x in txt.replace("-", ",").split(",") if x)


def tick_us():
    return 1e6 / int(subprocess.check_output(["getconf", "CLK_TCK"], text=True))


def core_ticks(cpu):
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu%d " % cpu):
                v = [int(x) for x in line.split()[1:]]
                return v[3] + (v[4] if len(v) > 4 else 0), sum(v)
    raise RuntimeError("cpu%d missing" % cpu)


def scope_stat():
    sp = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip() \
        .splitlines()[0].split("::")[-1]
    d = {}
    for line in open(sp + "/cpu.stat"):
        k, v = line.split()
        if k in ("nr_periods", "nr_throttled", "throttled_usec", "usage_usec"):
            d[k] = int(v)
    return sp, d


def workload():
    s = 0
    for i in range(WORKLOAD_ITERS):
        s += i * i
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True, type=int)
    ap.add_argument("--window", required=True, type=int)
    a = ap.parse_args()

    sibs = siblings(a.cpu)
    tu = tick_us()
    sp, st0 = scope_stat()
    mask0 = sorted(os.sched_getaffinity(0))

    ticks0 = {c: core_ticks(c) for c in sibs}
    t0 = time.time()
    durs = []
    for _ in range(RUNS):
        x = time.perf_counter()
        workload()
        durs.append((time.perf_counter() - x) * 1e6)
    t1 = time.time()
    time.sleep(max(0.0, SPAN_S - (t1 - t0)))
    ticks1 = {c: core_ticks(c) for c in sibs}
    sp2, st1 = scope_stat()
    span_us = (time.time() - t0) * 1e6
    mask1 = sorted(os.sched_getaffinity(0))

    per_sib = {}
    busy_total = 0.0
    for c in sibs:
        (i0, t_0), (i1, t_1) = ticks0[c], ticks1[c]
        busy = ((t_1 - t_0) - (i1 - i0)) * tu
        busy_total += busy
        per_sib[str(c)] = {"busy_us": round(busy, 1),
                           "util": round(busy / span_us, 4) if span_us else None}
    ours = st1["usage_usec"] - st0["usage_usec"]
    other_us = busy_total - ours
    other_util = round(other_us / span_us, 4) if span_us else None

    spread = spread_us(durs)
    rec = {
        "window": a.window,
        "pinned_cpu": a.cpu,
        "siblings": sibs,
        "mask_before": mask0, "mask_after": mask1,
        "mask_valid_pinned": mask0 == [a.cpu] == mask1,
        "tick_us": tu, "span_us": round(span_us, 1),
        "per_sibling": per_sib,
        "core_busy_us_both": round(busy_total, 1),
        "our_scope_usage_us": ours,
        "other_cpu_us": round(other_us, 1),
        "other_utilisation": other_util,
        "other_util_below_threshold": (other_util is not None
                                       and other_util < UTIL_THRESHOLD),
        "threshold": UTIL_THRESHOLD,
        "run_durations_us": [round(d, 1) for d in durs],
        "spread_us": round(spread, 1),
        "spread_within_requirement": spread <= REQUIREMENT_US,
        "requirement_us": REQUIREMENT_US,
        "workload_iters": WORKLOAD_ITERS, "runs": RUNS,
        "timestamp_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "timestamp_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t1)),
        "scope": sp,
        "throttled_fraction_delta": ((st1["nr_throttled"] - st0["nr_throttled"]) /
                                     max(1, st1["nr_periods"] - st0["nr_periods"])),
        "context_nondecisional": {"loadavg": open("/proc/loadavg").read().split()[:3]},
    }
    with open(os.path.join(HERE, "session_%d.json" % a.window), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
