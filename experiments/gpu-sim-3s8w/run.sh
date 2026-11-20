#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$(git rev-parse --show-toplevel)"
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_gate_stage_report.py -q
