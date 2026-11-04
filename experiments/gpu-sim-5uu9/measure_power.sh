#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD="$HERE/build"
rm -rf "$BUILD"; mkdir -p "$BUILD"
python3 "$HERE/build_powerarm.py" --build-dir "$BUILD"
for rep in 1 2 3; do
  python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" --build-dir "$BUILD" --out "$HERE/power_rep${rep}.jsonl"
done
echo "MEASURE_DONE"
