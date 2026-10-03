#!/usr/bin/env bash
# gpu-sim-u2o.6.6.6.2.1 -- CPU only, run directly. Never through ts: a CPU job in
# the shared GPU queue blocks real GPU work for its whole runtime, and nothing
# here touches the device.
#
# Re-derives the retro statement from the branch and the clock on every run, so
# if 6.7.1's rows are ever regenerated on a cached path this FAILS rather than
# remaining true. Nonzero exit on any failed property or test.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-u2o.6.6.6.2.1"
echo "=== retro statement: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 "$ART/measure_retro.py" 2>&1 | tee "$ART/output.log"
rc=${PIPESTATUS[0]}; echo "=== measure_retro exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"
echo "=== test suite ==="
python3 tests/test_retro_rows_uncached.py 2>&1 | tee -a "$ART/output.log" | tail -4
rc=${PIPESTATUS[0]}; echo "=== tests exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"
echo "=== run.sh: all green ==="
