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
clean = [s for s in sessions if not s["quiescence"]["contaminated"]]
contaminated = [s for s in sessions if s["quiescence"]["contaminated"]]
clean_spreads = sorted(s["spread_us"] for s in clean)
if len(clean) >= 2:
    clean_median = clean_spreads[len(clean_spreads)//2]
    admits = (sum(1 for s in clean if s["spread_us"] <= REQUIREMENT_US) >= 2) and (clean_median <= REQUIREMENT_US)
else:
    clean_median = None
    admits = False

if contaminated and not clean:
    verdict = "CONTAMINATED-ONLY"
elif admits:
    verdict = "ADMITTED"
else:
    verdict = "NO-ADMISSION-QUIESCED"

metrics = {
    "bead": "gpu-sim-s9j",
    "question": "Does the spread fall under 36.2us when our own activity is excluded (8-CPU quota)?",
    "pre_registration": "experiments/gpu-sim-s9j/pre_registration.md",
    "pre_registration_timestamp_utc": "2026-10-21T13:20:00Z",
    "first_measurement_timestamp_utc": min(s["timestamp_start_utc"] for s in sessions),
    "requirement_us": REQUIREMENT_US,
    "threshold_quiescence_us": 50000,
    "sessions": sessions,
    "clean_sessions": [s["session"] for s in clean],
    "contaminated_sessions": [s["session"] for s in contaminated],
    "clean_spreads_us": clean_spreads,
    "clean_median_us": clean_median,
    "verdict": verdict,
    "verdict_detail": "%s: %d/%d windows contaminated (other_cpu_us ~%.0f-%.0f us vs 50000 us threshold). "
                      "No clean window available for admission test." % (
        verdict, len(contaminated), len(sessions),
        min(s["quiescence"]["other_cpu_us"] for s in contaminated),
        max(s["quiescence"]["other_cpu_us"] for s in contaminated)),
}

out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "metrics.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print("wrote", out_path, "verdict =", verdict)
