#!/bin/bash
set -euo pipefail
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-mxaz
OUT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-mxaz/experiments/gpu-sim-mxaz
PY=/home/gabe/pi-research-lab/bin/python
$PY $OUT/measure_act.py --dtype bfloat16 --n-layers 8 --d 1024 --d-ff 4096 --n-heads 8 --steps 5 --warmup 2 > $OUT/raw_bf16.log 2>&1
echo "BF16 DONE"
$PY $OUT/measure_act.py --dtype float32 --n-layers 8 --d 1024 --d-ff 4096 --n-heads 8 --steps 5 --warmup 2 > $OUT/raw_fp32.log 2>&1
echo "FP32 DONE"
echo "ALL DONE"
