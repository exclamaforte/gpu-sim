#!/usr/bin/env python3
"""gpu-sim-r27: per-run SMT instrument.

Samples the SMT sibling's /proc/stat counters around EACH individual run (~6ms), so
interference is attributed to the run it happened in, and records WHO occupies the
sibling (processes allowed on that CPU, classified OUR-LAB vs EXTERNAL by cgroup).

Run: systemd-run --user --scope --property=CPUQuota=800% -- taskset -c 7 python3 run.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us  # noqa: E402

TARGET = 7
NWINDOWS = 12
WORKLOAD_ITERS = 100000
RUNS = 4          # run 0 is warmup and is dropped from the per-run analysis
SPAN_S = 0.200
REQUIREMENT_US = 36.2
RHO_THRESHOLD = 0.5
LO, HI = 0.25, 0.45
MIN_PER_GROUP = 4
# this lab's scopes live under
LAB_MARKERS = ("gpu-sim",)


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


def cgroup_of(pid):
    try:
        return open("/proc/%d/cgroup" % pid).read().strip().split("::")[-1]
    except Exception:
        return "?"


def allowed_on(pid, cpu):
    try:
        for line in open("/proc/%d/status" % pid):
            if line.startswith("Cpus_allowed_list"):
                spec = line.split(":")[1].strip()
                allowed = set()
                for part in spec.split(","):
                    if "-" in part:
                        a, b = part.split("-")
                        allowed.update(range(int(a), int(b) + 1))
                    elif part:
                        allowed.add(int(part))
                return cpu in allowed
    except Exception:
        pass
    return False


def who_occupies(sib, n=3):
    """Top CPU consumers whose affinity includes the sibling."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,pcpu,comm", "--sort=-pcpu"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    res = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, pcpu, comm = int(parts[0]), float(parts[1]), parts[2]
        except ValueError:
            continue
        if not allowed_on(pid, sib):
            continue
        cg = cgroup_of(pid)
        res.append({"pid": pid, "pcpu": pcpu, "comm": comm, "cgroup": cg,
                    "class": "OUR-LAB" if any(m in cg for m in LAB_MARKERS)
                    else "EXTERNAL"})
        if len(res) >= n:
            break
    return res


def workload():
    s = 0
    for i in range(WORKLOAD_ITERS):
        s += i * i
    return s


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
    n = len(xs)
    if n < 4:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else None


SIB = sibling_of(TARGET)


def main():
    tu = tick_us()
    windows = []
    for w in range(1, NWINDOWS + 1):
        mask = sorted(os.sched_getaffinity(0))
        runs = []
        t0 = time.time()
        for r in range(RUNS):
            i0, t0s = cpu_ticks(SIB)
            a = time.perf_counter()
            workload()
            dur = (time.perf_counter() - a) * 1e6
            i1, t1s = cpu_ticks(SIB)
            sib_ticks = (t1s - t0s) - (i1 - i0)
            runs.append({
                "run": r,
                "duration_us": round(dur, 1),
                "sibling_busy_ticks": sib_ticks,
                "sibling_util_during_run": round(sib_ticks * tu / dur, 4),
            })
        t1 = time.time()
        time.sleep(max(0.0, SPAN_S - (t1 - t0)))
        rec = {
            "window": w,
            "pinned_cpu": TARGET,
            "sibling": SIB,
            "mask_valid_pinned": mask == [TARGET],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "runs": runs,
            "spread_us": round(spread_us([r["duration_us"] for r in runs]), 1),
            "who_occupies_sibling": who_occupies(SIB, 3),
            "loadavg": open("/proc/loadavg").read().split()[:3],
        }
        windows.append(rec)
        with open(os.path.join(HERE, "windows.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("window %2d  spread %8.1f  runs=%s"
              % (w, rec["spread_us"],
                 [(round(x["duration_us"]), x["sibling_util_during_run"])
                  for x in runs]), flush=True)

    valid = [w for w in windows if w["mask_valid_pinned"]]
    pairs = [(r["sibling_util_during_run"], r["duration_us"])
             for w in valid for r in w["runs"][1:]]      # drop each window's warmup
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n_lo = sum(1 for v in xs if v < LO)
    n_hi = sum(1 for v in xs if v > HI)
    rho = spearman(xs, ys)

    if len(valid) < NWINDOWS:
        verdict = "PASS-INVALID"
    elif n_lo < MIN_PER_GROUP or n_hi < MIN_PER_GROUP:
        verdict = "RANGE-INSUFFICIENT"
    elif rho is not None and rho >= RHO_THRESHOLD:
        verdict = "ASSOCIATION-SUPPORTED"
    else:
        verdict = "ASSOCIATION-NOT-SUPPORTED"

    lo_d = [y for x, y in pairs if x < LO]
    hi_d = [y for x, y in pairs if x > HI]
    occ = [o for w in valid for o in w["who_occupies_sibling"]]
    our = sum(1 for o in occ if o["class"] == "OUR-LAB")

    res = {
        "bead": "gpu-sim-r27",
        "executor": "PI",
        "pre_registration": "experiments/gpu-sim-r27/pre_registration.md",
        "target_cpu": TARGET, "sibling_cpu": SIB,
        "n_windows": len(windows), "n_valid": len(valid),
        "n_runs_analysed": len(pairs),
        "spearman_rho_per_run": round(rho, 4) if rho is not None else None,
        "rho_threshold": RHO_THRESHOLD,
        "n_runs_sibling_util_below_0_25": n_lo,
        "n_runs_sibling_util_above_0_45": n_hi,
        "mean_duration_low_group_us": round(sum(lo_d) / len(lo_d), 1) if lo_d else None,
        "mean_duration_high_group_us": round(sum(hi_d) / len(hi_d), 1) if hi_d else None,
        "occupants": {
            "n_snapshots": len(occ),
            "n_our_lab": our,
            "n_external": len(occ) - our,
            "top": occ[:6],
        },
        "verdict": verdict,
        "no_admission_claimed": True,
        "prior_verdicts_unchanged": {
            "y1h": "RANGE-INSUFFICIENT",
            "52i": "NO-BOTH-QUIET-CORE-IN-20-SAMPLES",
            "99g": "NO-ADMISSION-CORRECTED-INSTRUMENT",
            "3ba": "NO-ADMISSION-QUIET-CORE",
        },
    }
    json.dump(res, open(os.path.join(HERE, "metrics.json"), "w"), indent=2)
    json.dump({"verdict": verdict}, open(os.path.join(HERE, "verdict.json"), "w"))
    print(json.dumps({k: res[k] for k in
                      ("verdict", "spearman_rho_per_run",
                       "n_runs_sibling_util_below_0_25",
                       "n_runs_sibling_util_above_0_45",
                       "mean_duration_low_group_us",
                       "mean_duration_high_group_us",
                       "occupants")}, indent=2)[:1200])


if __name__ == "__main__":
    main()
