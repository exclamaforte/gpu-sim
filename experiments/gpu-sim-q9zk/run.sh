#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/home/gabe/Documents/gpu-sim"
LAB="${ROOT}/.lab"
WT="${LAB}/wt"
BUILD="${WT}/gpu-sim-qaiu/experiments/gpu-sim-u5r4/build_div_sizescale"
OUT="$HERE/measured_rows_sizescale_strict.jsonl"
python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" \
  --build-dir "$BUILD" \
  --out "$OUT" --strict
