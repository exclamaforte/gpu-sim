#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export U5R4_OUT="${U5R4_OUT:-$ROOT/.lab/tmp/nlgf_out}"
case "$U5R4_OUT" in
    "$ROOT"/.lab/tmp/*) ;;
    *) echo "REFUSE: U5R4_OUT must be under $ROOT/.lab/tmp (got '$U5R4_OUT'); refusing to rm -rf it" >&2; exit 2 ;;
esac
rm -rf "$U5R4_OUT"; mkdir -p "$U5R4_OUT"
python3 "$HERE/analyze_power.py"
if ! diff -q "$U5R4_OUT/metrics.json" "$HERE/metrics.json" >/dev/null; then
    echo "METRICS_MISMATCH"; diff "$U5R4_OUT/metrics.json" "$HERE/metrics.json"; exit 1
fi
echo "METRICS_MATCH"
