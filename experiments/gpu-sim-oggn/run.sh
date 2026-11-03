#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/../.."
python3 scripts/microbatch_sweep.py >/dev/null
python3 -m pytest tests/test_microbatch_sweep.py -q 2>&1 | tail -1
