#!/usr/bin/env bash
# gpu-sim-zhq -- recompute check. CPU-ONLY (stdlib).
# Re-runs the latency-arm tests + before/after driver into scratch,
# byte-compares metrics.json. Writes no tracked file. NOTE: unlike g8x's
# gate, a byte-identical before/after here is a FAILURE (AC1) -- the driver
# exits nonzero and this script reports RECOMPUTE FAIL (UNCHANGED).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
SCRATCH="$(mktemp -d /tmp/zhq-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 -m unittest tests.test_latency_arm tests.test_roofline_specs 2>&1 | tail -n 3
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/zhq_verify.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
tail -n 1 "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: driver verdict=fail (predictions unchanged?)"; exit 1; }
cmp -s "$HERE/metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical, verdict rc=$rc)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
