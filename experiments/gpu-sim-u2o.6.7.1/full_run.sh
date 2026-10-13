#!/usr/bin/env bash
# gpu-sim-u2o.6.7.1 cold-fix full confirm -- recompute check. CPU-ONLY: the GPU
# dose (full 264, default cold) cannot be recomputed without a device. Re-runs
# full_verify.py into scratch and byte-compares against the committed
# full_metrics.json (deterministic: committed rows + git-show reference).
# Writes nothing tracked. Fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/coldfull-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/full_verify.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: analyzer exited $rc"; exit 1; }
cmp -s "$HERE/full_metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
