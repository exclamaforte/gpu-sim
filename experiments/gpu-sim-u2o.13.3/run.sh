#!/bin/bash
# gpu-sim-u2o.13.3 replay: search tests + smoke (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.13.3
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_surrogate_133.py -q -p no:cacheprovider || exit 1
python3 epic_u2o/surrogate_search.py || exit 1
echo "REPLAY-OK"
