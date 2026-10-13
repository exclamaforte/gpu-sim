#!/usr/bin/env bash
# gpu-sim-edu.1 -- CPU only. Run directly, never through ts: a CPU job in the
# shared GPU queue blocks real GPU work for its whole runtime, and nothing here
# touches the device.
#
# Recomputes metrics.json from committed code and runs the suite. Exits nonzero
# if any acceptance property fails or any test fails.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."   # repo root of this worktree
ART="experiments/gpu-sim-edu.1"

echo "=== dispatch metrics: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 "$ART/measure_dispatch.py" 2>&1 | tee "$ART/output.log"
rc=${PIPESTATUS[0]}
echo "=== measure_dispatch exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== filtered dispatch, with reasons ==="
python3 scripts/bd_ready_filtered.py --explain 2>&1 | tee -a "$ART/output.log" | head -20
rc=${PIPESTATUS[0]}
echo "=== filter exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== test suite ==="
python3 tests/test_bd_ready_filter.py 2>&1 | tee -a "$ART/output.log" | tail -5
rc=${PIPESTATUS[0]}
echo "=== tests exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== run.sh: all green ==="