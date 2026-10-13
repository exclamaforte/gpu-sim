#!/bin/bash
# gpu-sim-u2o.11.1 replay: overfit proof (~4-5 min CPU).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.11.1
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_overfit_111.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
