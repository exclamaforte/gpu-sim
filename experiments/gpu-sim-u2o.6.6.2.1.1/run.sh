#!/usr/bin/env bash
# gpu-sim-u2o.6.6.2.1.1 -- CPU only, run directly. Never through ts.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-u2o.6.6.2.1.1"
echo "=== supersession inventory: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 "$ART/measure_supersession.py" 2>&1 | tee "$ART/output.log"
rc=${PIPESTATUS[0]}; echo "=== measure exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"
echo "=== test suite ==="
python3 tests/test_tree_state_supersession.py 2>&1 | tee -a "$ART/output.log" | tail -4
rc=${PIPESTATUS[0]}; echo "=== tests exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"
echo "=== run.sh: all green ==="
