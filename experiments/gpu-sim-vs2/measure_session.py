#!/usr/bin/env python3
"""Measure one session under the current cgroup quota. CPU-only.

Usage:
  python3 measure_session.py --setting S1 --session 1
  (for S2, invoke under: systemd-run --user --scope --property=CPUQuota=800% --
     python3 measure_session.py --setting S2 --session 1)

Writes session_<setting>_<session>.json in this directory.
"""
import argparse, json, os, time, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us
WORKLOAD_ITERS = 100000
RUNS = 4  # 1 warmup + 3 measured


def workload():
    s = 0
    for i in range(WORKLOAD_ITERS):
        s += i * i
    return s


def read_stat():
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
    """Snapshot of top CPU-consuming processes not in our cgroup, via ps."""
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
    ap.add_argument("--setting", required=True, choices=["S1", "S2"])
    ap.add_argument("--session", required=True, type=int)
    args = ap.parse_args()

    aff = read_affinity()
    per_cpu_util = cpu_util(aff, 1.0)
    cg_before, stat_before, quota, period = read_stat()
    top_consumers = top_non_cgroup_consumers(cg_before, 3)
    ts_start = time.time()
    durations = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        workload()
        durations.append((time.perf_counter() - t0) * 1e6)
    ts_end = time.time()
    cg_after, stat_after, _, _ = read_stat()

    spread = spread_us(durations)
    delta = {k: stat_after[k] - stat_before[k] for k in stat_before}

    record = {
        "setting": args.setting,
        "session": args.session,
        "timestamp_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_start)),
        "timestamp_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_end)),
        "duration_s": round(ts_end - ts_start, 3),
        "cpu_max": "%s %s" % (quota if quota is not None else "max", period),
        "quota_cpus": (quota / period) if (quota and period) else None,
        "affinity_mask": aff,
        "per_cpu_util": per_cpu_util,
        "mean_affinity_util": round(sum(per_cpu_util.values())/len(per_cpu_util), 4),
        "top3_non_cgroup_consumers": top_consumers,
        "cgroup_scope": cg_before,
        "run_durations_us": [round(d, 1) for d in durations],
        "spread_us": round(spread, 1),
        "workload_iters": WORKLOAD_ITERS,
        "runs": RUNS,
        "cpu_stat_before": stat_before,
        "cpu_stat_after": stat_after,
        "cpu_stat_delta": delta,
        "throttled_fraction_delta": (delta["nr_throttled"] / delta["nr_periods"]
                                     if delta.get("nr_periods", 0) > 0 else None),
    }
    out = os.path.join(HERE, "session_%s_%d.json" % (args.setting, args.session))
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
