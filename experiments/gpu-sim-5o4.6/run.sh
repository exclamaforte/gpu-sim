#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/../.."
python -c "import json; d=json.load(open('experiments/gpu-sim-5o4.6/metrics.json')); assert 'N8' in d and 'N16' in d"
python -m pytest tests/test_topology_search.py -q
echo "OK"
