#!/usr/bin/env python3
"""Build metrics.json from session_*.json. CPU-only."""
import json, os, sys, glob
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us, PROXY_TIMER_ARTIFACT_US

sessions = []
for p in sorted(glob.glob(os.path.join(HERE, "session_*.json"))):
    sessions.append(json.load(open(p)))

REQUIREMENT_US = 36.2
TIGHT_DELTA_US = 362.3

s1 = [s for s in sessions if s["setting"] == "S1"]
s2 = [s for s in sessions if s["setting"] == "S2"]
s1_spreads = sorted(s["spread_us"] for s in s1)
s2_spreads = sorted(s["spread_us"] for s in s2)
s2_admit_count = sum(1 for s in s2 if s["spread_us"] <= REQUIREMENT_US)
s2_median = s2_spreads[len(s2_spreads)//2] if s2_spreads else None
s2_admits = (s2_admit_count >= 2) and (s2_median is not None and s2_median <= REQUIREMENT_US)

verdict = "ADMITTED-AT-S2" if s2_admits else "NOT-A-QUOTA-EFFECT"

metrics = {
    "bead": "gpu-sim-vs2",
    "question": "Does an admissible operating point exist at a larger (8-CPU) quota?",
    "pre_registration": "experiments/gpu-sim-vs2/pre_registration.md",
    "pre_registration_timestamp_utc": "2026-10-21T12:55:03Z",
    "first_measurement_timestamp_utc": min(s["timestamp_start_utc"] for s in sessions),
    "requirement_us": REQUIREMENT_US,
    "tightest_modelled_delta_us": TIGHT_DELTA_US,
    "settings": {
        "S1": {"quota": "200000 100000 (2 CPUs)", "sessions": len(s1)},
        "S2": {"quota": "800000 100000 (8 CPUs)", "sessions": len(s2),
               "obtained_via": "systemd-run --user --scope --property=CPUQuota=800% -- <cmd>"},
    },
    "interleaving_order": "S1_1, S2_1, S1_2, S2_2, S1_3, S2_3",
    "sessions": sessions,
    "summary": {
        "S1_spreads_us": s1_spreads,
        "S1_median_us": s1_spreads[len(s1_spreads)//2],
        "S2_spreads_us": s2_spreads,
        "S2_median_us": s2_median,
        "S2_sessions_admitting": s2_admit_count,
        "S2_sessions_total": len(s2),
    },
    "decision_rule": "S2 ADMITS iff spread <= 36.2 us in >=2 of 3 sessions AND median spread <= 36.2 us.",
    "verdict": verdict,
    "verdict_detail": "%s: S2 spreads %s us, median %.1f us; %d/%d sessions <= 36.2 us." % (
        verdict, s2_spreads, s2_median, s2_admit_count, len(s2)),
    "prediction_check": "PI pre-registered prediction was NOT-A-QUOTA-EFFECT. "
                        "Result: %s. Prediction %s." % (verdict,
                        "CORRECT" if verdict == "NOT-A-QUOTA-EFFECT" else "WRONG (state plainly)"),
}

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "metrics.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("wrote", out_path, "verdict =", verdict)
