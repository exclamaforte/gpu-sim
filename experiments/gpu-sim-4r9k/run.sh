#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - << 'PY'
import json, sys
sys.path.insert(0, "experiments/gpu-sim-8zqr")
from measure import nearest_residual
rows=[json.loads(l) for l in open("experiments/gpu-sim-4r9k/measured_gemm.jsonl")]
usable=0
for r in rows:
    n=r["size"]; residual=nearest_residual(n)
    within=r["within_std_us"]; kernel_us=r["kernel_us"]; grid=r["grid"]
    md=r.get("metadata",{}) if isinstance(r.get("metadata"),dict) else {}
    compute_bound = bool(md.get("compute_bound", False)) or (r.get("family") in ("gemm_naive",))
    # threshold (mirrors measure.py:92-102)
    threshold = 5*max(within, residual) if (within>0 or residual>0) else 0.0
    if compute_bound:
        threshold = 5 * residual if residual > 0 else threshold
    # grid_capped (per-dimension, mirrors measure.py + test_grid_capped_boundary)
    gc = (grid[0] > (2**31 - 1)) or (grid[1] > 65535) or (grid[2] > 65535)
    if kernel_us >= threshold and (not gc) and kernel_us > 0:
        usable += 1
print(f"usable after fix: {usable}/{len(rows)}")
assert usable == 24, f"expected 24, got {usable}"
print("OK")
PY

# --- Regenerate decision table and deep-compare against committed ---
python3 experiments/gpu-sim-4r9k/gen_decision_table.py

# --- Assert regenerated artifacts carry rejection_reason (regenerated, not pre-tip) ---
python3 - << 'PY'
import json
p = "experiments/gpu-sim-8zqr/powered_rows.jsonl"
rows = [json.loads(l) for l in open(p) if l.strip()]
assert rows and "rejection_reason" in rows[0], "powered_rows.jsonl is pre-tip (no rejection_reason); regenerate or document"
print(f"OK: powered_rows.jsonl regenerated with rejection_reason ({len(rows)} rows)")
PY
