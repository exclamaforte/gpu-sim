#!/usr/bin/env bash
# gpu-sim-u2o.6.6.7 -- CPU only. Run directly, never through ts: a CPU job in the
# shared GPU queue blocks real GPU work for its whole runtime, and nothing here
# touches the device.
#
# Recomputes metrics.json from committed code and runs the suite. Exits nonzero
# if any property fails or any test fails.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-u2o.6.6.7"

echo "=== spread + verdicts: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 "$ART/measure_spread.py" 2>&1 | tee "$ART/output.log"
rc=${PIPESTATUS[0]}
echo "=== measure_spread exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== test suite ==="
python3 tests/test_replicate_discipline.py 2>&1 | tee -a "$ART/output.log" | tail -5
rc=${PIPESTATUS[0]}
echo "=== tests exit=$rc ==="
[ "$rc" -eq 0 ] || exit "$rc"

echo "=== run.sh: all green ==="