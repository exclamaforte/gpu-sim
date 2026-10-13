#!/usr/bin/env bash
# gpu-sim-u2o.6.2 recovery exercise -- recompute check. CPU-ONLY: the GPU dose
# (induced hang + survivor) cannot be recomputed without a device. Re-runs
# recover_analyze.py into scratch and byte-compares against the committed
# recover_metrics.json (deterministic: committed steps + recover_job.json).
# Writes nothing tracked. Fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/u62rec-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/recover_analyze.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
# rc is the VERDICT (0=pass, 1=fail), not a crash code: a fail verdict still
# recomputes byte-identically. Only a crash (rc 2, no metrics written) fails.
[ "$rc" -eq 0 ] || [ "$rc" -eq 1 ] || { echo "RECOMPUTE FAIL: analyzer crashed ($rc)"; exit 1; }
 HISTORY="$HERE/recover_metrics.json"
cmp -s "$HISTORY" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical, verdict rc=$rc)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
