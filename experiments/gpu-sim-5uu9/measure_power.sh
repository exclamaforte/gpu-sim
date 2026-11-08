#!/bin/bash
# NOTE: this script writes power_rep{1,2,3}.jsonl (tracked) in place into the experiment dir.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/.lab/tmp/5uu9_power_build}"
rm -rf "$BUILD"; mkdir -p "$BUILD"
python3 "$HERE/build_powerarm.py" --build-dir "$BUILD"
for rep in 1 2 3; do
  python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" --build-dir "$BUILD" --out "$HERE/power_rep${rep}.jsonl"
done
echo "MEASURE_DONE"
