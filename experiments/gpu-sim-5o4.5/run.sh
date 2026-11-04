#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/../.."
python experiments/gpu-sim-5o4.5/sensitivity.py >/tmp/5o4.5.out
python -c "import json; d=json.load(open('experiments/gpu-sim-5o4.5/metrics.json')); assert 'Not separable at this margin' in d['verdict']"
echo "OK"
