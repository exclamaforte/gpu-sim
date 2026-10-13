#!/usr/bin/env bash
# gpu-sim-39u EXPERIMENT 2 -- recompute check. CPU-ONLY (stdlib).
# Re-runs exp2_nlocal.py into scratch, byte-compares metrics.json.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
SCRATCH="$(mktemp -d /tmp/exp2-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/exp2_nlocal.py" \
    --metrics-out "$SCRATCH/recomputed.json" 2>&1 | tail -n 4
cmp -s "$HERE/metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
