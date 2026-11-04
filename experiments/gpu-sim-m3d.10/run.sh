#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/../.."
OUT="$(pwd)/experiments/gpu-sim-m3d.10"
mkdir -p "$OUT"
make check > "$OUT/make_check.log" 2>&1
cd "$OUT"
python3 admissibility.py > "$OUT/admissibility_run.log" 2>&1
echo "DONE" >> "$OUT/admissibility_run.log"
