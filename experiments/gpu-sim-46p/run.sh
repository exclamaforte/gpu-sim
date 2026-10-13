#!/usr/bin/env bash
# gpu-sim-46p -- recompute check. CPU-ONLY (stdlib + repo).
# Re-runs the unit gates + before/after driver, byte-compares metrics.json.
# The driver has no change-detection gate (46p EXPECTS movement, stated
# pre-run); the artifact carries the preregistered expectation instead.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 -m unittest \
    tests.test_flat_memory_arm tests.test_roofline_specs 2>&1 | tail -n 3
SCRATCH="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/46p-recompute-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$HERE/metrics.json" "$SCRATCH/metrics.committed"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/bw_verify.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
tail -n 6 "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: driver rc=$rc"; exit 1; }
cmp -s "$SCRATCH/metrics.committed" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
