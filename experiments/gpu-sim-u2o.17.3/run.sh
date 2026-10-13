#!/bin/bash
# gpu-sim-u2o.17.3 replay: scaling-law fit + checks (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.17.3
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_sim_trace_173.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
