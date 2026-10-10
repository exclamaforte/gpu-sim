#!/usr/bin/env bash
# gpu-sim-e5q -- recompute check. CPU-ONLY: re-runs e5q_verify.py into scratch
# and byte-compares metrics.json + job.json. Writes no tracked file.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/e5q-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
cp metrics.json "$SCRATCH/committed_metrics.json"
cp job.json "$SCRATCH/committed_job.json"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/e5q_verify.py" \
    --metrics-out "$SCRATCH/recomputed.json" \
    --job-out "$SCRATCH/recomputed_job.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || [ "$rc" -eq 1 ] || { echo "RECOMPUTE FAIL: crashed ($rc)"; exit 1; }
cmp -s "$SCRATCH/committed_metrics.json" "$SCRATCH/recomputed.json" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
cmp -s "$SCRATCH/committed_job.json" "$SCRATCH/recomputed_job.json" \
    && echo "RECOMPUTE PASS (byte-identical metrics+job, verdict rc=$rc)" \
    || { echo "RECOMPUTE FAIL: job differs"; exit 1; }
