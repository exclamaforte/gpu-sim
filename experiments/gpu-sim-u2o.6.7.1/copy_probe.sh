#!/usr/bin/env bash
# gpu-sim-u2o.6.7.1 copy-locus causal probe -- GPU DOSE (one ts job, three arms).
# Hypothesis: batched-worker warmth -- same-process allocation reuse / cache-TLB
# warmth makes later copy cells faster than cold single-run medians.
# Arms (same 40-trial protocol, per-example sampling, copy-only subset):
#   warm    -- one persistent process; position_in_batch 0..32 recorded
#   cold    -- buffers freed+reallocated between examples (warm-state control)
#   perproc -- independent process per invocation (7.1's isolation; coldest)
# Each arm asserts device-time-unchanged (protocol integrity); a moved device
# time fails the job rather than producing a number. Outputs to OUTDIR.
set -uo pipefail
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.6.7.1-c/experiments/gpu-sim-u2o.6.7.1
OUTDIR=${U2O_COPY_OUTDIR:-/home/gabe/Documents/gpu-sim-runs/u671-copyprobe-20261006}
mkdir -p "$OUTDIR"
cd "$WT"
for mode in warm cold perproc; do
  echo "=== arm: $mode ==="
  python3 batch_measure_worker.py --mode "$mode" --op-filter copy \
      --out "$OUTDIR/rows_copy_${mode}.jsonl" \
      --summary-out "$OUTDIR/summary_copy_${mode}.json"
  echo "--- device gate: $mode ---"
  python3 - "$OUTDIR/summary_copy_${mode}.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
d = s["device_time_per_example"]
print(f"measured={d['measured_us']:.1f}us asserted={d['asserted_us']:.1f} unchanged={d['unchanged']}")
sys.exit(0 if d["unchanged"] else 1)
EOF
done
echo "COPY PROBE COMPLETE: $(ls "$OUTDIR"/rows_copy_*.jsonl | wc -l) arms in $OUTDIR"
