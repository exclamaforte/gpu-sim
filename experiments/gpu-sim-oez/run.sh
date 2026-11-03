#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/../.."
test -f experiments/gpu-sim-oez/decision.md
test -f experiments/gpu-sim-oez/metrics.json
python3 -c "import json; d=json.load(open('experiments/gpu-sim-oez/metrics.json')); assert d['decision']=='b'"
echo "OK"
