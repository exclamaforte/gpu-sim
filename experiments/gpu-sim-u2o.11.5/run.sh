#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# gpu-sim-u2o.11.5 replay: writeup verification (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.11.5
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_writeup_115.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
