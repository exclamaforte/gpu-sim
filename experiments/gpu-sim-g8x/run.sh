#!/usr/bin/env bash
# gpu-sim-g8x -- recompute check. CPU-ONLY (stdlib; no numpy needed).
# Re-runs the specs tests + baseline driver into scratch, byte-compares
# metrics.json. Writes no tracked file.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
SCRATCH="$(mktemp -d /tmp/g8x-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 -m unittest tests.test_roofline_specs 2>&1 | tail -n 3
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/baseline_rerun.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
tail -n 1 "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || exit 1
cmp -s "$HERE/metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical, verdict rc=$rc)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
