#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# CPU-side validation: recompute the 405B floors from the committed measured bytes/param
# and check they match experiments/gpu-sim-37xk/measured_opt_bytes.json.
python3 - <<'PY'
import json
m = json.load(open("experiments/gpu-sim-37xk/measured_opt_bytes.json"))
# recomputed floors
fp32_fused = 81.2 * (m["measured_bytes_per_param"]["fp32_moments"]["measured_peak_fused"] /
                     m["measured_bytes_per_param"]["fp32_moments"]["assumed_peak_fused"])
assert abs(fp32_fused - m["recomputed_floors_gb"]["fp32_moments_fused"]) < 0.15, (fp32_fused, m["recomputed_floors_gb"]["fp32_moments_fused"])
print("fp32 fused floor (measured):", round(fp32_fused, 1), "GB")
print("OK")
PY