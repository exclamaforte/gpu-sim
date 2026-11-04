#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${TMPDIR:-/tmp}/gpu-sim-bwn-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
python3 "$HERE/run_bwn.py" >/dev/null
python3 "$HERE/compute_metrics.py" --out "$SCRATCH/metrics.json"
diff -u "$HERE/metrics.json" "$SCRATCH/metrics.json" && echo "metrics.json matches recomputed output"