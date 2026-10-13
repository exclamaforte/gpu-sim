#!/usr/bin/env bash
# gpu-sim-u2o.6.4 (READ-ONLY HALF) -- CPU/diagnostic only.
#
# NOT a measurement. `nvidia-smi --query-gpu` is read-only DIAGNOSTIC telemetry,
# permitted outside the queue by the 2026-09-27 rule amendment and never
# citable as a measurement. It performs no device work and cannot contaminate
# another tenant's measurement.
#
# The MUTATING half of this bead -- -lgc/-pl and persistence-mode setup -- is
# HELD PENDING A HUMAN DECISION and is never invoked here.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-u2o.6.4"
echo "=== read-only telemetry: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 "$ART/gpu_telemetry.py" --samples 12 --interval 0.4 2>&1 | tee "$ART/output.log"
rc=${PIPESTATUS[0]}; echo "=== telemetry exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"
echo "=== tests ==="
python3 tests/test_gpu_telemetry.py 2>&1 | tee -a "$ART/output.log" | tail -4
rc2=${PIPESTATUS[0]}; echo "=== tests exit=$rc2 ==="; [ "$rc2" -eq 0 ] || exit "$rc2"
echo "=== run.sh: all green ==="
