#!/usr/bin/env bash
# re02: reproduce the gate budget measurement.
#
# Fast path (what a drain runs): prose-vs-committed checks only, target <=240s.
# Full path (what a land runs): adds the artifact-vs-code recompute, which is
# the part that costs minutes (error_budget.py alone ~2-4 min of CPU).
#
# Both are recorded here so the claim in metrics.json can be re-derived rather
# than trusted: the previous revision of this file ran only the fast path, so
# the expensive half of the check was never exercised at land time.
set -euo pipefail
cd "$(dirname "$0")/../.."

echo "=== fast path (CI_GATES_FULL=0) ==="
t0=$(date +%s)
python3 scripts/ci_gates.py || true
t1=$(date +%s)
echo "fast_wall_s=$((t1-t0))"

# The full GATE is `bash scripts/gate.sh` (no 290s cap) and runs at land.
# What is measured here is the expensive CHECK the split moved behind the env
# var -- running it under ci_gates' 290s cap would only re-measure the cap.
echo "=== full check (CI_GATES_FULL=1): artifact-vs-code recompute ==="
t0=$(date +%s)
CI_GATES_FULL=1 python3 scripts/check_docs_model_numbers.py
t1=$(date +%s)
echo "full_check_wall_s=$((t1-t0))"
