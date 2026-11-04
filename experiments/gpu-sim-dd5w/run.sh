#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m unittest tests.test_7uz2_raw_to_facts tests.test_3fog_raw_to_facts tests.test_37xk_raw_to_facts
