#!/usr/bin/env python3
"""gpu-sim-99g: measure ONE pinned window with the CORRECTED admissibility input.

Corrected instrument (see pre_registration.md):
  other_cpu_us = (core_total_ticks_delta - core_idle_ticks_delta) * tick_us
                 - our_scope_cpu_us
  other_utilisation = other_cpu_us / span_us
where our_scope_cpu_us is THIS cgroup's cpu.stat usage_usec delta (the measuring
scope only, never the user slice) and tick_us = 1e6 / getconf CLK_TCK.

Must be invoked under `taskset -c <cpu>`. Writes session_<n>.json.
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
SPAN_S = 0.200              # span containing the window (pre-registered)
UTIL_THRESHOLD = 0.25       # unchanged
REQUIREMENT_US = 36.2       # unchanged


def tick_us():
    hz = int(subprocess.check_output(["getconf", "CLK_TCK"], text=True).strip())
    return 1e6 / hz


def core_ticks(cpu):
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu%d " % cpu):
                vals = [int(x) for x in line.split()[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                return idle, sum(vals)
    raise RuntimeError("cpu%d missing from /proc/stat" % cpu)


def scope_path():
    return "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip() \
        .splitlines()[0].split("::")[-1]


def scope_stat(sp):
    d = {}
    for line in open(sp + "/cpu.stat"):
        k, v = line.split()
        if k in ("nr_periods", "nr_throttled", "throttled_usec", "usage_usec"):
            d[k] = int(v)
    return d


def workload():
    s = 0
    for i in range(WORKLOAD_ITERS):
        s += i * i
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True, type=int)
    ap.add_argument("--window", required=True, type=int)
    args = ap.parse_args()

    tu = tick_us()
    sp = scope_path()
    mask_before = sorted(os.sched_getaffinity(0))

    idle0, tot0 = core_ticks(args.cpu)
    st0 = scope_stat(sp)
    t_start = time.time()
    durations = []
    for _ in range(RUNS):
        a = time.perf_counter()
        workload()
        durations.append((time.perf_counter() - a) * 1e6)
    t_end = time.time()
    time.sleep(max(0.0, SPAN_S - (t_end - t_start)))
    idle1, tot1 = core_ticks(args.cpu)
    st1 = scope_stat(sp)
    span_s = time.time() - t_start
    mask_after = sorted(os.sched_getaffinity(0))

    core_busy_us = (tot1 - tot0) * tu - (idle1 - idle0) * tu
    ours_us = st1["usage_usec"] - st0["usage_usec"]
    span_us = span_s * 1e6
    other_us = core_busy_us - ours_us
    other_util = round(other_us / span_us, 4) if span_us > 0 else None

    spread = spread_us(durations)
    delta = {k: st1[k] - st0[k] for k in st0}
    rec = {
        "window": args.window,
        "target_cpu": args.cpu,
        "mask_before": mask_before,
        "mask_after": mask_after,
        "mask_valid_pinned": mask_before == [args.cpu] == mask_after,
        "tick_us": tu,
        "span_s": round(span_s, 4),
        "core_ticks_delta": tot1 - tot0,
        "core_idle_ticks_delta": idle1 - idle0,
        "core_busy_us": round(core_busy_us, 1),
        "our_scope_usage_us": ours_us,
        "other_cpu_us": round(other_us, 1),
        "other_utilisation": other_util,
        "other_util_below_threshold": (other_util is not None
                                       and other_util < UTIL_THRESHOLD),
        "threshold": UTIL_THRESHOLD,
        "run_durations_us": [round(d, 1) for d in durations],
        "spread_us": round(spread, 1),
        "spread_within_requirement": spread <= REQUIREMENT_US,
        "requirement_us": REQUIREMENT_US,
        "workload_iters": WORKLOAD_ITERS,
        "runs": RUNS,
        "timestamp_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime(t_start)),
        "timestamp_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime(t_end)),
        "scope": sp,
        "throttled_fraction_delta": (delta["nr_throttled"] / delta["nr_periods"]
                                     if delta.get("nr_periods", 0) > 0 else None),
        "context_nondecisional": {
            "loadavg": open("/proc/loadavg").read().split()[:3],
            "note": "context only; may NOT admit or exclude a window",
        },
    }
    out = os.path.join(HERE, "session_%d.json" % args.window)
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
