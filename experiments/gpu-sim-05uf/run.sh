#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/check_docs_model_numbers.py
CI_GATES_FULL=1 python scripts/check_docs_model_numbers.py
python -m pytest tests/test_act_factor_knob.py -q
