#!/usr/bin/env bash
# gpu-sim-u2o.6.7.1 cold-fix verify -- GPU DOSE 1 of 2 (copy subset).
# Runs the worker in DEFAULT mode (no --mode flag: cold since the copy
# verdict) on the 33 copy cells. Asserts device-time-unchanged; a moved
# device time fails the job rather than producing a number.
set -uo pipefail
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.6.7.1-d/experiments/gpu-sim-u2o.6.7.1
OUTDIR=${U2O_FIX_OUTDIR:-/home/gabe/Documents/gpu-sim-runs/u671-coldfix-20261006}
mkdir -p "$OUTDIR"
cd "$WT"
python3 batch_measure_worker.py --op-filter copy \
    --out "$OUTDIR/rows_copy_fix.jsonl" \
    --summary-out "$OUTDIR/summary_copy_fix.json"
python3 - "$OUTDIR/summary_copy_fix.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
d = s["device_time_per_example"]
print(f"mode={s['mode']} measured={d['measured_us']:.1f}us asserted={d['asserted_us']:.1f} unchanged={d['unchanged']}")
sys.exit(0 if (d["unchanged"] and s["mode"] == "cold") else 1)
EOF
echo "FIX VERIFY COMPLETE in $OUTDIR"
