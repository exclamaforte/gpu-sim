#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="$(pwd)/experiments/gpu-sim-m3d.10"
mkdir -p "$OUT"
make check > "$OUT/make_check.log" 2>&1
cd "$OUT"
python3 admissibility.py > "$OUT/admissibility_run.log" 2>&1
echo "DONE" >> "$OUT/admissibility_run.log"
