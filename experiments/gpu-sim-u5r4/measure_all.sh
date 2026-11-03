#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$HERE/measured_rows.jsonl"
# Resume path: measure.py appends and skips already-measured (family,build_idx) keys.
# Do NOT rm -f here -- a kill/timeout must not discard progress.
python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" --build-dir "$HERE/build_gemm" --out "$OUT"
python3 "$ROOT/experiments/gpu-sim-8zqr/measure.py" --build-dir "$HERE/build_div" --out "$OUT"
echo "MEASURE_DONE rows=$(wc -l < "$OUT")"
