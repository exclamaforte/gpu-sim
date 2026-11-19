#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$(git rev-parse --show-toplevel)"
scratch="$(pwd)/.lab/tmp/mve4_run.out"
mkdir -p "$(dirname "$scratch")"
CUDA_VISIBLE_DEVICES= python3 -m pytest \
  tests/test_k9e4_registry_superset.py \
  tests/test_k9e4_pin_loadbearing.py \
  tests/test_e7yw_trunk_resolution.py -q >"$scratch" 2>&1
rc=$?
cat "$scratch"
exit $rc
