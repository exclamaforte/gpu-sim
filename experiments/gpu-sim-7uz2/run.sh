#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/gen_7uz2_metrics.py
python scripts/check_docs_model_numbers.py
python -m unittest tests.test_7uz2_act_curve
