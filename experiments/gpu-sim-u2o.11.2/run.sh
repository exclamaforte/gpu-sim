#!/bin/bash
# gpu-sim-u2o.11.2 replay: CPU inference (~2 min) + analysis checks.
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.11.2
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 experiments/gpu-sim-u2o.11.2/infer_112.py "$WT/experiments/gpu-sim-u2o.11.2" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_failure_modes_112.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
