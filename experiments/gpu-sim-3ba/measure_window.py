#!/usr/bin/env python3
"""Measure one pinned window (gpu-sim-ep3 §4-§6). Must be invoked under
taskset -c <target_cpu>.

Writes session_<n>.json. Verifies the affinity mask INSIDE the measurement: a
window whose mask is not exactly {target} is marked INVALID (ep3 §4).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us  # noqa: E402

WORKLOAD_ITERS = 100000
RUNS = 4                      # 1 warmup + 3 measured
UTIL_SPAN_S = 0.200           # ep3 §5: 20 ticks at USER_HZ=100
REQUIREMENT_US = 36.2
UTIL_THRESHOLD = 0.25


def workload():
    s = 0
    for i in range(WORKLOAD_ITERS):
        s += i * i
    return s


def cpu_stat(cpu):
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu%d " % cpu):
                vals = [int(x) for x in line.split()[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                return idle, sum(vals)
    raise RuntimeError("cpu%d not in /proc/stat" % cpu)


def cgroup_stat():
    cg = open("/proc/self/cgroup").read().strip().splitlines()[0].split("::")[-1]
    base = "/sys/fs/cgroup" + cg
    d = {}
    for line in open(base + "/cpu.stat"):
        k, v = line.split()
        if k in ("nr_periods", "nr_throttled", "throttled_usec", "usage_usec"):
            d[k] = int(v)
    try:
        q, per = open(base + "/cpu.max").read().split()
        quota = None if q == "max" else int(q)
        period = int(per)
    except Exception:
        quota, period = None, None
    return cg, d, quota, period


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", required=True, type=int)
    ap.add_argument("--window", required=True, type=int)
    args = ap.parse_args()

    mask_before = sorted(os.sched_getaffinity(0))
    valid = mask_before == [args.cpu]

    cg, stat_before, quota, period = cgroup_stat()

    # Utilisation of the target CPU over a span CONTAINING the window (ep3 §5).
    i0, t0s = cpu_stat(args.cpu)
    ts_start = time.time()
    durations = []
    for _ in range(RUNS):
        a = time.perf_counter()
        workload()
        durations.append((time.perf_counter() - a) * 1e6)
    ts_end = time.time()
    time.sleep(max(0.0, UTIL_SPAN_S - (ts_end - ts_start)))
    i1, t1s = cpu_stat(args.cpu)

    _, stat_after, _, _ = cgroup_stat()
    mask_after = sorted(os.sched_getaffinity(0))
    valid = valid and mask_after == [args.cpu]

    util_span = round(1.0 - ((i1 - i0) / (t1s - t0s)), 4) if t1s > t0s else None
    spread = spread_us(durations)
    delta = {k: stat_after[k] - stat_before[k] for k in stat_before}

    rec = {
        "window": args.window,
        "target_cpu": args.cpu,
        "quota": "%s %s" % (quota if quota is not None else "max", period),
        "quota_cpus": (quota / period) if (quota and period) else None,
        "cgroup_scope": cg,
        "affinity_mask_before": mask_before,
        "affinity_mask_after": mask_after,
        "mask_valid_pinned": valid,
        "util_span_s": UTIL_SPAN_S,
        "target_cpu_util_window_span": util_span,
        "target_cpu_util_below_25pct": (util_span is not None
                                        and util_span < UTIL_THRESHOLD),
        "run_durations_us": [round(d, 1) for d in durations],
        "spread_us": round(spread, 1),
        "spread_within_requirement": spread <= REQUIREMENT_US,
        "requirement_us": REQUIREMENT_US,
        "workload_iters": WORKLOAD_ITERS,
        "runs": RUNS,
        "timestamp_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime(ts_start)),
        "timestamp_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime(ts_end)),
        "duration_s": round(ts_end - ts_start, 4),
        "cpu_stat_delta": delta,
        "throttled_fraction_delta": (delta["nr_throttled"] / delta["nr_periods"]
                                     if delta.get("nr_periods", 0) > 0 else None),
        "context_nondecisional": {
            "loadavg": open("/proc/loadavg").read().split()[:3],
            "note": "context only; MAY NOT admit or exclude a window (ep3 §7)",
        },
    }
    out = os.path.join(HERE, "session_%d.json" % args.window)
    with open(out, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
