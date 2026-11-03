#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS="${U5R4_METRICS:-$HERE/metrics.json}"
SCRATCH="${TMPDIR:-$HERE/../../.lab/tmp}/gpu-sim-u5r4-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
U5R4_OUT="$SCRATCH" python3 "$HERE/analyze.py" > "$SCRATCH/analyze.log" 2>&1
echo "compared: $SCRATCH/metrics.json  vs  $METRICS"
if ! diff -u "$SCRATCH/metrics.json" "$METRICS"; then
  echo "metrics.json NOT reproducible"; exit 1; fi
echo "metrics.json reproducible"
echo "OK: u5r4 analysis reproducible"
