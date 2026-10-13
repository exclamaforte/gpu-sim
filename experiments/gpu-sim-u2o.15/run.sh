#!/usr/bin/env bash
# gpu-sim-u2o.15 -- CPU only, run directly. Never through ts.
# Audits every bead the fabrication audit reopened, and fails if any has been
# closed again without a CONCRETE artifact pointer.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-u2o.15"
echo "=== close audit: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/close_requires_artifact.py 2>&1 | tee "$ART/output.log"
rc=${PIPESTATUS[0]}; echo "=== audit exit=$rc ==="
echo "=== tests ==="
python3 tests/test_close_requires_artifact.py 2>&1 | tee -a "$ART/output.log" | tail -4
rc2=${PIPESTATUS[0]}; echo "=== tests exit=$rc2 ==="
[ "$rc" -eq 0 ] && [ "$rc2" -eq 0 ] || exit 1
echo "=== run.sh: all green ==="
