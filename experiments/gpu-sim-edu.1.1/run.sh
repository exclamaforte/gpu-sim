#!/usr/bin/env bash
# gpu-sim-edu.1.1 -- recompute check. CPU-ONLY (fixture repos live in /tmp).
# Re-runs edu11_verify.py into scratch and byte-compares against committed
# edu11_metrics.json. Writes no tracked file (suite log goes to scratch).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/edu11-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/edu11_verify.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || [ "$rc" -eq 1 ] || { echo "RECOMPUTE FAIL: crashed ($rc)"; exit 1; }
cmp -s "$HERE/edu11_metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical, verdict rc=$rc)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; diff "$HERE/edu11_metrics.json" "$SCRATCH/recomputed.json" | head -n 10; exit 1; }
