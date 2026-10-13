#!/bin/bash
# gpu-sim-u2o.11.6 replay: script tests (no live-db mutation) + board read.
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.11.6
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_bd_exp_116.py -q -p no:cacheprovider || exit 1
bash scripts/bd-exp.sh board | grep -q "gpu-sim-u2o.10.4.1" || exit 1
echo "REPLAY-OK"
