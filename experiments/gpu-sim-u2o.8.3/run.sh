#!/usr/bin/env bash
# gpu-sim-u2o.8.3 -- recompute check. CPU-ONLY (stdlib + repo).
# Re-runs the unit gates + populate, byte-compares both artifacts against
# the committed ones. A same-vector/different-runtime pair, a zero
# bytes_moved, or a hash-unrecoverable row fails the DRIVER (nonzero exit),
# which fails this script. Writes no tracked file (outputs restored after
# comparison).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 -m unittest \
    tests.test_feature_vector tests.test_latency_arm tests.test_roofline_specs \
    2>&1 | tail -n 3
SCRATCH="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/u83-recompute-XXXXXX")"
# NOTE: not /tmp -- its user quota breaks on features.jsonl-sized copies.
trap 'rm -rf "$SCRATCH"' EXIT
cp "$HERE/metrics.json" "$SCRATCH/metrics.committed"
cp "$HERE/features.jsonl" "$SCRATCH/features.committed"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/populate_feature_vector.py" \
    > "$SCRATCH/out.log" 2>&1
rc=$?
tail -n 5 "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: populate verdict=FAIL"; exit 1; }
cmp -s "$SCRATCH/metrics.committed" "$HERE/metrics.json" \
    || { echo "RECOMPUTE FAIL: metrics.json differs"; exit 1; }
cmp -s "$SCRATCH/features.committed" "$HERE/features.jsonl" \
    || { echo "RECOMPUTE FAIL: features.jsonl differs"; exit 1; }
echo "RECOMPUTE PASS (byte-identical metrics.json + features.jsonl)"
