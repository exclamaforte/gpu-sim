#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/measured_rows_v2.jsonl"
# resume-safe: do NOT rm -f
python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" --build-dir "$HERE/build_gemm_v2" --out "$OUT"
python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" --build-dir "$HERE/build_div_v2" --out "$OUT"
echo "MEASURE_DONE rows=$(wc -l < "$OUT")"