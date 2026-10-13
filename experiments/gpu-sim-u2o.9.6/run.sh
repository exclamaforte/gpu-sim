#!/bin/bash
# gpu-sim-u2o.9.6 replay: shared-loop proof on real data (CPU-only, ~1 min).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.9.6
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_training_loop_96.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
