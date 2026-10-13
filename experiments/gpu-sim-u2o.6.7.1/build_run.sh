#!/usr/bin/env bash
# gpu-sim-u2o.6.7.1 build packet -- recompute check. CPU-ONLY: the GPU dose
# (ts 20261006t062332-cfcbaa4f) cannot be recomputed without a device.
# Re-runs build_metrics.py into scratch and byte-compares against the
# committed build_metrics.json. Writes nothing tracked. Fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/u671build-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/build_metrics.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: analyzer exited $rc"; exit 1; }
cmp -s "$HERE/build_metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
