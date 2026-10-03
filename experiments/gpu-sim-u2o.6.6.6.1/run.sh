#!/usr/bin/env bash
# gpu-sim-u2o.6.6.6.1 rework -- CPU only, runs directly (never through ts:
# a CPU job in the shared GPU queue blocks real GPU work for its whole
# runtime, and nothing here touches the device).
#
# Recomputes spot_check.json from committed code and runs the full suite.
# Fails if any gate or any test fails. Exact millisecond figures are NOT
# asserted across runs (they move with box load); gate outcomes are.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."   # repo root of this worktree

echo "=== spot check: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 "experiments/gpu-sim-u2o.6.6.6.1/measure_cost.py" 2>&1 | tee "experiments/gpu-sim-u2o.6.6.6.1/output.log"
rc=${PIPESTATUS[0]}
echo "=== measure_cost exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== cost suite ==="
python3 tests/test_queue_state_cost.py 2>&1 | tee -a "experiments/gpu-sim-u2o.6.6.6.1/output.log"
rc=${PIPESTATUS[0]}
echo "=== test_queue_state_cost exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== differential suite ==="
python3 tests/test_cache_differential.py 2>&1 | tee -a "experiments/gpu-sim-u2o.6.6.6.1/output.log"
rc=${PIPESTATUS[0]}
echo "=== test_cache_differential exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== run.sh: all green ==="
