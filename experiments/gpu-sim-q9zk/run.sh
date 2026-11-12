#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/measured_rows_sizescale_strict.jsonl"
python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" \
  --build-dir "$HERE/../../gpu-sim-qaiu/experiments/gpu-sim-u5r4/build_div_sizescale" \
  --out "$OUT" --strict
