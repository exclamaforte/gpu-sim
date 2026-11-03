#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "GPU measurement job: 20261028t234659-d45ecb2c"
echo "Results in raw_bf16.log and raw_fp32.log"
