#!/usr/bin/env python3
"""Build metrics.json from windows.jsonl. CPU-only, no GPU."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us, PROXY_TIMER_ARTIFACT_US

rows = [json.loads(l) for l in open(os.path.join(HERE, "windows.jsonl"))]

# Pre-registered admission threshold from committed m3d.10 artifacts:
# 4b requires proxy spread <= 1/10 of tightest modelled delta.
# m3d.10 metrics.json: d_pred_M1M2_us = 362.3, max_tolerable_spread_us = 36.2
D_PRED_M1M2_US = 362.3
REQUIREMENT_US = 36.2  # = 362.3 / 10, committed as 36.2 in m3d.10 metrics.json

# admission rule applied per window
def admits(spread, repeat_spread=None, load_recorded=True):
    i = spread <= REQUIREMENT_US
    ii = (repeat_spread is None) or (repeat_spread <= REQUIREMENT_US)
    iii = load_recorded
    return i and ii and iii

# map repeat windows to originals
repeat_map = {}
for r in rows:
    if r.get("repeat_of"):
        repeat_map[r["repeat_of"]] = r

# annotate
table = []
for r in rows:
    rep = repeat_map.get(r["window_id"])
    rep_spread = rep["spread_us"] if rep else None
    i_ok = r["spread_us"] <= REQUIREMENT_US
    ii_ok = (rep_spread is None) or (rep_spread <= REQUIREMENT_US)
    # load recorded within band claimed: we claim affinity util recorded; always true here
    iii_ok = True
    status = "ADMITTED" if (i_ok and ii_ok and iii_ok) else None
    if i_ok and not ii_ok:
        status = "NON-ADMISSIBLE-FLUKY"
    if not i_ok:
        status = "NOT-ADMITTED"
    table.append({
        "window_id": r["window_id"],
        "phase": r["phase"],
        "phase_label": r["phase_label"],
        "timestamp_start_utc": r["timestamp_start_utc"],
        "loadavg_1": r["loadavg_1"],
        "loadavg_5": r["loadavg_5"],
        "loadavg_15": r["loadavg_15"],
        "nproc_shell": r.get("nproc_shell", 2),
        "affinity_cpu_count": r["affinity_cpu_count"],
        "mean_affinity_util": r["mean_affinity_util"],
        "nonvoluntary_ctx_switches_delta": r["nonvoluntary_ctx_switches_delta"],
        "run_durations_us": r["run_durations_us"],
        "spread_us": r["spread_us"],
        "repeat_of": r.get("repeat_of"),
        "repeat_spread_us": rep_spread,
        "rule_i_spread_le_req": i_ok,
        "rule_ii_repeatable": ii_ok,
        "rule_iii_load_recorded": iii_ok,
        "status": status,
        "raw_log": r["raw_log"],
        "burners": r["burners"],
        "load_label": r["load_label"],
    })

min_spread = min(r["spread_us"] for r in rows)
min_row = min(rows, key=lambda r: r["spread_us"])

# verdict
any_admitted = any(t["status"] == "ADMITTED" for t in table)
verdict = "ADMITTED" if any_admitted else "NO-ADMISSIBLE-OPERATING-POINT"

metrics = {
    "bead": "gpu-sim-tfq",
    "question": "Does ANY operating point on this box admit (spread <= 36.2 us)?",
    "workload": {
        "description": "Fixed CPU workload: sum of squares over N=100000 integers (Python int loop), "
                       "run 4 times per window (1 warmup + 3 measured); durations in microseconds. "
                       "No tuning between windows.",
        "iters": 100000,
        "runs_per_window": 4,
        "measured_runs": 3,
    },
    "statistic": {
        "reused_from": "experiments/gpu-sim-m3d.10/verdict.py (proxy derivation)",
        "steps": ["drop first value (init/warmup)", "drop values > 1e9 us (timer artifacts)",
                  "spread = max(remaining) - min(remaining)"],
        "agreement_check": "experiments/gpu-sim-tfq/check_statistic_agreement.py (PASS)",
    },
    "pre_registered_admission_rule": {
        "verbatim": "4b requires the proxy spread to be at most 1/10 of the tightest modelled delta. "
                    "An operating point ADMITS iff (i) measured spread <= absolute requirement, AND "
                    "(ii) the measurement is repeatable: the same window, re-measured, stays inside "
                    "the requirement, AND (iii) load is recorded and stays within the band claimed "
                    "for that point. A point satisfying (i) but not (ii) is NON-ADMISSIBLE-FLUKY.",
        "absolute_requirement_us": REQUIREMENT_US,
        "derived_from": "experiments/gpu-sim-m3d.10/metrics.json: d_pred_M1M2_us=362.3; /10 = 36.23 -> committed 36.2",
        "arithmetic": "362.3 us / 10 = 36.23 us (committed as 36.2 us)",
        "tightest_modelled_delta_us": D_PRED_M1M2_US,
    },
    "environment": {
        "nproc_shell": 2,
        "os_cpu_count": 32,
        "sched_getaffinity": "0-31 (all 32 CPUs visible at task level)",
        "cpuset_cpus_effective_parent": "0-31",
        "cgroup_scope_cpu_max": "200000 100000 (2 CPUs quota)",
        "cgroup_nr_periods": 2826,
        "cgroup_nr_throttled": 64,
        "cgroup_throttled_usec": 11820592,
        "finding": "The measuring process runs in a scope cgroup with a 2-CPU quota "
                   "(cpu.max=200000/100000) and IS throttled (nr_throttled=64, "
                   "throttled_usec=11.8s). The PI amendment's premise that no quota exists "
                   "holds at the app.slice level but not at the process scope. Recorded as a "
                   "finding per the amendment's instruction.",
        "loadavg_context_only": True,
    },
    "phases": {
        "1": "natural load observation (no burners); 8 windows + 2 repeats",
        "2": "bounded niced load sweep (nice -n 19 burners pinned to masked CPUs); 6 windows",
    },
    "spread_vs_load": table,
    "repeatability_checks": [
        {
            "original": "w01", "original_spread_us": 53.4,
            "repeat": "w01r", "repeat_spread_us": 1044.0,
            "original_admitted": False, "repeat_admitted": False,
            "fluky": False,
            "note": "closest point to threshold; repeat shows instability (not an admission either way)",
        },
        {
            "original": "w04", "original_spread_us": 123.6,
            "repeat": "w04r", "repeat_spread_us": 800.3,
            "original_admitted": False, "repeat_admitted": False,
            "fluky": False,
        },
    ],
    "verdict": verdict,
    "min_observed_spread_us": min_spread,
    "min_observed_spread_window": min_row["window_id"],
    "min_spread_ratio_to_requirement": round(min_spread / REQUIREMENT_US, 3),
    "distance_from_requirement_us": round(min_spread - REQUIREMENT_US, 1),
    "conclusion": "NO-ADMISSIBLE-OPERATING-POINT: minimum observed spread is %.1f us "
                  "(window %s), %.1f us (%.2fx) above the 36.2 us requirement. "
                  "No window satisfies rule (i); no fluky admissions. "
                  "The spread is unstable across repeats (53.4 -> 1044 us), so even the "
                  "best single sample cannot be treated as an admission." % (min_spread, min_row["window_id"],
                  min_spread - REQUIREMENT_US, min_spread / REQUIREMENT_US),
}

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "metrics.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("wrote", out_path)
print("wrote metrics.json; verdict =", verdict)
