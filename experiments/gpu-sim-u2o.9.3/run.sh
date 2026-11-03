#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# gpu-sim-u2o.9.3 replay: loss unit tests only (CPU-only bead, no GPU used).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.9.3
cd "$WT" || exit 1
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_noise_loss.py \
    tests/test_trans_baseline.py tests/test_fusion.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
