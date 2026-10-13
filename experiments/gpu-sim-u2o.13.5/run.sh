#!/bin/bash
# gpu-sim-u2o.13.5 replay: acquisition tests + smoke (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.13.5
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_acquisition_135.py -q -p no:cacheprovider || exit 1
python3 epic_u2o/acquisition.py || exit 1
echo "REPLAY-OK"
