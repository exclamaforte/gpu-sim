#!/usr/bin/env bash
# gpu-sim-u2o.6.3 -- real-GPU validation of the resource budgets.
# Run ONLY through the shared ts queue (gpu-sim-u2o.6.8).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"

echo "=== ts job: $(hostname) at $(date -Is) ==="

echo "--- CPU unit tests (no GPU) ---"
python3 test_resource_limits.py > cpu_tests.log 2>&1
cpu_rc=$?
tail -3 cpu_tests.log
echo "cpu tests exit=$cpu_rc"

echo "--- GPU validation ---"
python3 verify_budgets.py 2>&1 | tee output.log
gpu_rc=${PIPESTATUS[0]}

echo "=== cpu_rc=$cpu_rc gpu_rc=$gpu_rc ==="
[ "$cpu_rc" -eq 0 ] && [ "$gpu_rc" -eq 0 ]
