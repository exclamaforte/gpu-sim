#!/bin/bash
# gpu-sim-u2o.17.4 replay: parity gate (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.17.4
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_parity_174.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
