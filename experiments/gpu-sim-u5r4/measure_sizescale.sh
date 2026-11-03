#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/measured_rows_sizescale.jsonl"
python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" --build-dir "$HERE/build_div_sizescale" --out "$OUT"
echo "MEASURE_DONE rows=$(wc -l < "$OUT")"
