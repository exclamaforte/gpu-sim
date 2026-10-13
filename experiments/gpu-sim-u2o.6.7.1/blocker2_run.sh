#!/usr/bin/env bash
# gpu-sim-u2o.6.7.1 blocker2-direction -- recompute check. CPU-ONLY (no GPU
# dose needed: cross-dataset agreement is the confirmation). Re-runs
# blocker2_direction.py into scratch and byte-compares against the committed
# blocker2_metrics.json (deterministic: committed rows + git-show reference).
# Writes nothing tracked. Fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/b2dir-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/blocker2_direction.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: analyzer exited $rc"; exit 1; }
cmp -s "$HERE/blocker2_metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
