#!/bin/bash
# gpu-sim-u2o.13.1 replay: gate check (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.13.1
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_gate_131.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
