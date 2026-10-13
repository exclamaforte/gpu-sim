#!/usr/bin/env bash
# gpu-sim-u2o.6.6.6.3.1 -- CPU only. Run directly, never through ts: a CPU job
# in the shared GPU queue blocks real GPU work, and nothing here touches the
# device (fixture queues in /tmp plus read-only timing of the live queue dir).
#
# Runs the retry-bound suite (deterministic: sleep counts, not wall time) and
# the pre-existing queue_state suites that can run in a worktree. Exits
# nonzero if anything fails.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-u2o.6.6.6.3.1"

echo "=== retry-bound suite: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 tests/test_queue_retry_bound.py 2>&1 | tee "$ART/output.log"
rc=${PIPESTATUS[0]}
echo "=== retry-bound exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== pre-existing queue_state suites ==="
python3 -m unittest tests.test_cache_differential tests.test_queue_state \
    tests.test_queue_state_cost tests.test_step_queue_state 2>&1 \
    | tee -a "$ART/output.log" | tail -3
rc=${PIPESTATUS[0]}
echo "=== suites exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== run.sh: all green ==="
