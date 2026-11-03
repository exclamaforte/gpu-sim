#!/usr/bin/env bash
# gpu-sim-rzg: model-size sweep (CPU only, no GPU). Re-runs the sweep and writes
# experiments/gpu-sim-rzg/sweep.json. sweep.py asserts the 8B/N8 row still equals
# the audited gpu-sim-qk9 answer (66.99 ms, tp=1 pp=1 cp=1 dp=8 zero=2).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/../.."
python3 experiments/gpu-sim-rzg/sweep.py
