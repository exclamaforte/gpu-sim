#!/bin/bash
# gpu-sim-u2o.10.2 replay: ranking tests incl. real-data run (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.10.2
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_ranking_102.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
