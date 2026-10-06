#!/usr/bin/env bash
# gpu-sim-u2o.6.7.1 cold-fix FULL-CORPUS confirm -- GPU DOSE 2 of 2.
# Runs the worker in DEFAULT mode (cold) on the full 264 sweep. The pass bar
# is PRE-REGISTERED in full_verify.py BEFORE this run (PI ruling 16:51):
# stable cold-vs-cold movers baselined out, edge flips adjudicated by
# same-condition repeat, bar as a number. Asserts mode==cold +
# device-time-unchanged in-job; a violation fails the job.
set -uo pipefail
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.6.7.1-d/experiments/gpu-sim-u2o.6.7.1
OUTDIR=${U2O_FULL_OUTDIR:-/home/gabe/Documents/gpu-sim-runs/u671-coldfull-20261006}
mkdir -p "$OUTDIR"
cd "$WT"
python3 batch_measure_worker.py \
    --out "$OUTDIR/rows_full_fix.jsonl" \
    --summary-out "$OUTDIR/summary_full_fix.json"
python3 - "$OUTDIR/summary_full_fix.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
d = s["device_time_per_example"]
print(f"mode={s['mode']} measured={d['measured_us']:.1f}us asserted={d['asserted_us']:.1f} unchanged={d['unchanged']}")
sys.exit(0 if (d["unchanged"] and s["mode"] == "cold") else 1)
EOF
echo "FULL CONFIRM COMPLETE in $OUTDIR"
