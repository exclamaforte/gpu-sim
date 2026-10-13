#!/bin/bash
# gpu-sim-u2o.10.1 replay: core-metrics tests incl. real-data run (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.10.1
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_core_metrics_101.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
