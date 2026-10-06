#!/usr/bin/env bash
# gpu-sim-u2o.6.6 dose -- recompute check. CPU-ONLY: the GPU dose
# (ts 20261006t041800-5878994f) cannot be recomputed without a device.
# Re-runs dose_analyze.py into scratch and byte-compares against the
# committed dose_metrics.json (deterministic: sort_keys, no RNG, no walls
# in the compared payload beyond recorded observations). Writes nothing
# tracked. Fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/u66dose-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/dose_analyze.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: analyzer exited $rc"; exit 1; }
cmp -s "$HERE/dose_metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
