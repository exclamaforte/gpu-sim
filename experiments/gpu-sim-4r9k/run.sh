#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/../.."
python3 - << 'PY'
import json, sys
sys.path.insert(0, "experiments/gpu-sim-8zqr")
from measure import gate_decision, nearest_residual
rows=[json.loads(l) for l in open("experiments/gpu-sim-4r9k/measured_gemm.jsonl")]
usable=0
for r in rows:
    u, thr, gc, reason = gate_decision(r)
    if u:
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

# --- Regenerate AC3 table and deep-compare against committed ---
python3 experiments/gpu-sim-4r9k/gen_ac3.py

# --- AC3 plant test ---
python3 -m pytest tests/test_ac3_plant.py -q
