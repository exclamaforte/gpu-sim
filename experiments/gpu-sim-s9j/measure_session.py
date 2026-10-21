#!/usr/bin/env python3
"""Measure one quiesced window at 8-CPU quota. CPU-only."""
import argparse, json, os, time, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us
WORKLOAD_ITERS = 100000
RUNS = 4
QUIESCENCE_THRESHOLD_US = 50000  # 50 ms of CPU per window
USER_SLICE = "/sys/fs/cgroup/user.slice/user-1000.slice"


def workload():
    s = 0
    for i in range(WORKLOAD_ITERS):
        s += i * i
    return s


def read_scope_stat():
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
    return cg, base, d, quota, period


def read_user_slice_usage():
    for line in open(USER_SLICE + "/cpu.stat"):
        k, v = line.split()
        if k == "usage_usec":
            return int(v)
    return 0


def read_affinity():
    return sorted(os.sched_getaffinity(0))


def read_proc_stat(indices):
    res = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            toks = line.split()
            name = toks[0]
            if name == "cpu":
                continue
            idx = int(name[3:])
            if idx not in indices:
                continue
            vals = [int(x) for x in toks[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            res[idx] = (idle, total)
    return res


def cpu_util(indices, sample_s=1.0):
    before = read_proc_stat(indices)
    time.sleep(sample_s)
    after = read_proc_stat(indices)
    out = {}
    for idx in indices:
        ib, tb = before[idx]
        ia, ta = after[idx]
        dt = ta - tb
        di = ia - ib
        out[idx] = round(1.0 - (di / dt), 4) if dt > 0 else 0.0
    return out


def top_non_cgroup_consumers(our_cg, n=3):
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,pcpu,cgroup,comm", "--sort=-pcpu"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    res = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, pcpu, cgroup, comm = parts
        if our_cg in cgroup:
            continue
        try:
            res.append({"pid": int(pid), "pcpu": float(pcpu),
                        "cgroup": cgroup, "comm": comm})
        except ValueError:
            continue
        if len(res) >= n:
            break
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=int)
    args = ap.parse_args()

    aff = read_affinity()
    top_consumers = top_non_cgroup_consumers("", 3)  # placeholder; refined below
    # refine our_cg
    cg0, base0, stat0, quota0, period0 = read_scope_stat()
    top_consumers = top_non_cgroup_consumers(cg0, 3)

    # quiescence: sample user-slice + scope before
    user_before = read_user_slice_usage()
    cg_before, base_before, scope_stat_before, quota, period = read_scope_stat()
    ts_start = time.time()

    # per-CPU util over 1s (the window)
    per_cpu_util = cpu_util(aff, 1.0)

    # run workload 4x
    durations = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        workload()
        durations.append((time.perf_counter() - t0) * 1e6)

    ts_end = time.time()
    user_after = read_user_slice_usage()
    cg_after, base_after, scope_stat_after, _, _ = read_scope_stat()

    spread = spread_us(durations)
    scope_delta_us = scope_stat_after["usage_usec"] - scope_stat_before["usage_usec"]
    user_delta_us = user_after - user_before
    other_cpu_us = user_delta_us - scope_delta_us
    contaminated = other_cpu_us > QUIESCENCE_THRESHOLD_US

    scope_throttle_delta = {
        "nr_periods": scope_stat_after["nr_periods"] - scope_stat_before["nr_periods"],
        "nr_throttled": scope_stat_after["nr_throttled"] - scope_stat_before["nr_throttled"],
        "throttled_usec": scope_stat_after["throttled_usec"] - scope_stat_before["throttled_usec"],
    }

    record = {
        "session": args.session,
        "timestamp_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_start)),
        "timestamp_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_end)),
        "duration_s": round(ts_end - ts_start, 3),
        "quota": "%s %s" % (quota if quota is not None else "max", period),
        "quota_cpus": (quota / period) if (quota and period) else None,
        "cgroup_scope": cg_before,
        "affinity_mask": aff,
        "per_cpu_util": per_cpu_util,
        "mean_affinity_util": round(sum(per_cpu_util.values()) / len(per_cpu_util), 4),
        "run_durations_us": [round(d, 1) for d in durations],
        "spread_us": round(spread, 1),
        "workload_iters": WORKLOAD_ITERS,
        "runs": RUNS,
        "scope_throttle_delta": scope_throttle_delta,
        "quiescence": {
            "user_slice_path": USER_SLICE,
            "user_slice_cpu_delta_us": user_delta_us,
            "scope_cpu_delta_us": scope_delta_us,
            "other_cpu_us": other_cpu_us,
            "threshold_us": QUIESCENCE_THRESHOLD_US,
            "contaminated": contaminated,
        },
        "top3_non_cgroup_consumers": top_consumers,
    }
    out = os.path.join(HERE, "session_%d.json" % args.session)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
