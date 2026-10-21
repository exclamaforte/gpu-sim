#!/usr/bin/env python3
"""Measure one observation window. CPU-only.

Usage:
  python3 measure_window.py --phase 1 --window-id w01
  python3 measure_window.py --phase 2 --window-id p2a --burners 2 --load-label "burners=2"
  python3 measure_window.py --phase 1 --window-id w01r --repeat-of w01

Writes raw log experiments/gpu-sim-tfq/raw/<window_id>.log (one us value per line,
first is warmup/init) and appends a record to windows.jsonl.
"""
import argparse, json, os, subprocess, time, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RAWDIR = os.path.join(HERE, "raw")
WINDOWS = os.path.join(HERE, "windows.jsonl")
WORKLOAD_ITERS = 100000  # fixed before first measurement; do not change between windows
WARMUP_PLUS = 4  # 1 warmup (init) + 3 measured runs
UTIL_SAMPLE_S = 1.0


def workload():
    s = 0
    for i in range(WORKLOAD_ITERS):
        s += i * i
    return s


def read_loadavg():
    with open("/proc/loadavg") as f:
        parts = f.read().split()
    return float(parts[0]), float(parts[1]), float(parts[2])


def read_affinity():
    return sorted(os.sched_getaffinity(0))

def read_cgroup_quota():
    try:
        cg = open("/proc/self/cgroup").read().strip().splitlines()[0].split("::")[-1]
    except Exception:
        cg = ""
    base = "/sys/fs/cgroup" + cg
    quota = None
    period = None
    try:
        q, per = open(base + "/cpu.max").read().split()
        quota = int(q) if q != "max" else None
        period = int(per)
    except Exception:
        pass
    stat = {}
    try:
        for line in open(base + "/cpu.stat"):
            k, v = line.split()
            if k in ("nr_periods", "nr_throttled", "throttled_usec"):
                stat[k] = int(v)
    except Exception:
        pass
    return {"cgroup_scope": cg, "cpu_quota_us": quota, "cpu_period_us": period,
            "cpu_quota_cpus": (quota / period) if (quota and period) else None,
            "nr_periods": stat.get("nr_periods"), "nr_throttled": stat.get("nr_throttled"),
            "throttled_usec": stat.get("throttled_usec")}


def read_proc_stat(indices):
    # returns dict idx -> (idle, total) cumulative jiffies
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
            # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            total = sum(vals)
            res[idx] = (idle, total)
    return res


def cpu_util(indices):
    before = read_proc_stat(indices)
    time.sleep(UTIL_SAMPLE_S)
    after = read_proc_stat(indices)
    out = {}
    for idx in indices:
        ib, tb = before[idx]
        ia, ta = after[idx]
        dt = ta - tb
        di = ia - ib
        out[idx] = (1.0 - (di / dt)) if dt > 0 else 0.0
    return out


def read_nonvoluntary_ctx():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("nonvoluntary_ctxt_switches:"):
                return int(line.split()[1])
    return 0


def spawn_burners(n, indices):
    procs = []
    # pin burners round-robin to masked cpus
    for i in range(n):
        cpu = indices[i % len(indices)]
        p = subprocess.Popen(
            ["taskset", "-c", str(cpu), "python3", "-c",
             "while True:\n  pass\n"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # renice
        try:
            subprocess.run(["renice", "-n", "19", "-p", str(p.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        procs.append(p)
    return procs


def kill_burners(procs):
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["1", "2"])
    ap.add_argument("--window-id", required=True)
    ap.add_argument("--load-label", default=None)
    ap.add_argument("--burners", type=int, default=0)
    ap.add_argument("--repeat-of", default=None)
    args = ap.parse_args()

    os.makedirs(RAWDIR, exist_ok=True)
    aff = read_affinity()
    burners = []
    if args.burners > 0:
        burners = spawn_burners(args.burners, aff)
        time.sleep(2.0)  # let burners ramp

    ctx_before = read_nonvoluntary_ctx()
    util = cpu_util(aff)
    loadavg = read_loadavg()
    ts_start = time.time()

    # run workload WARMUP_PLUS times (1 warmup + 3 measured)
    durations = []
    for k in range(WARMUP_PLUS):
        t0 = time.perf_counter()
        workload()
        dt_us = (time.perf_counter() - t0) * 1e6
        durations.append(dt_us)

    ts_end = time.time()
    ctx_after = read_nonvoluntary_ctx()

    # write raw log
    raw_path = os.path.join(RAWDIR, args.window_id + ".log")
    with open(raw_path, "w") as f:
        for d in durations:
            f.write("%.1f\n" % d)

    # spread via reused statistic
    sys.path.insert(0, HERE)
    from spread import spread_us
    spread = spread_us(durations)

    kill_burners(burners)

    record = {
        "window_id": args.window_id,
        "phase": int(args.phase),
        "phase_label": "natural" if args.phase == "1" else "niced-sweep",
        "timestamp_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_start)),
        "timestamp_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts_end)),
        "load_label": args.load_label if args.load_label else ("natural" if args.phase == "1" else "burners=%d" % args.burners),
        "repeat_of": args.repeat_of,
        "nproc": os.cpu_count(),
        "nproc_shell": 2,
        "affinity_mask": aff,
        "affinity_cpu_count": len(aff),
        "cgroup_quota": read_cgroup_quota(),
        "loadavg_1": loadavg[0],
        "loadavg_5": loadavg[1],
        "loadavg_15": loadavg[2],
        "per_cpu_util": {str(k): round(v, 4) for k, v in util.items()},
        "mean_affinity_util": round(sum(util.values()) / len(util), 4),
        "nonvoluntary_ctx_switches_delta": ctx_after - ctx_before,
        "workload_iters": WORKLOAD_ITERS,
        "warmup_plus_measured": WARMUP_PLUS,
        "run_durations_us": [round(d, 1) for d in durations],
        "spread_us": round(spread, 1),
        "raw_log": "experiments/gpu-sim-tfq/raw/%s.log" % args.window_id,
        "burners": args.burners,
    }
    with open(WINDOWS, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
