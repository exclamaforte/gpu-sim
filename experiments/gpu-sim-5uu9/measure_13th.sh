#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD="$ROOT/.lab/tmp/5uu9_13th_build"
rm -rf "$BUILD"; mkdir -p "$BUILD"
python3 "$HERE/build_13th.py" --build-dir "$BUILD"
for rep in 1 2 3; do
  python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" --build-dir "$BUILD" --out "$HERE/mech_probe_rep${rep}.jsonl"
done
echo "MEASURE_13TH_DONE"
