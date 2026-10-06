#!/usr/bin/env bash
# gpu-sim-u2o.6.7.1 rerun -- recompute check. CPU-ONLY: the GPU rerun
# (ts 20261006t055612-834191c6) cannot be recomputed without a device.
# Re-runs compare_rerun.py into scratch and byte-compares against the
# committed rerun_metrics.json (deterministic: committed rows + git-show
# reference + sort_keys). Writes nothing tracked. Fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/u671rerun-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/compare_rerun.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: analyzer exited $rc"; exit 1; }
cmp -s "$HERE/rerun_metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
