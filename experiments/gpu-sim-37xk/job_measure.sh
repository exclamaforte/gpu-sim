#!/bin/bash
set -euo pipefail
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-37xk
OUT=experiments/gpu-sim-37xk
/home/gabe/pi-research-lab/bin/python3 $OUT/measure_opt_bytes.py --dtype bfloat16 --n-params 200000000 --steps 8 > $OUT/raw_bf16.log 2>&1
/home/gabe/pi-research-lab/bin/python3 $OUT/measure_opt_bytes.py --dtype float32 --n-params 200000000 --steps 8 > $OUT/raw_fp32.log 2>&1
echo "DONE"