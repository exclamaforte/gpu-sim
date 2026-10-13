#!/bin/bash
# gpu-sim-u2o.10.4 replay: join + drift guard (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.10.4
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_compare_104.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
