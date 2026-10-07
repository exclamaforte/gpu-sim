#!/usr/bin/env bash
# gpu-sim-edu activation + regression cover -- recompute check. CPU-ONLY
# (no GPU involved; the live probe already ran). Re-runs edu_verify.py into
# scratch and byte-compares against the committed edu_metrics.json.
# Writes nothing tracked except suite_output.log inside the experiment dir
# (overwritten deterministically each run; the committed copy is the record).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/edurec-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
export EDU_SUITE_LOG="$SCRATCH/suite.log"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/edu_verify.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || [ "$rc" -eq 1 ] || { echo "RECOMPUTE FAIL: crashed ($rc)"; exit 1; }
cmp -s "$HERE/edu_metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical, verdict rc=$rc)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
