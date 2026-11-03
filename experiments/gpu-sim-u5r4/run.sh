#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${TMPDIR:-$HERE/../../.lab/tmp}/gpu-sim-u5r4-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
# Recompute analysis into scratch (never overwrite committed metrics.json).
U5R4_OUT="$SCRATCH" python3 "$HERE/analyze.py" > "$SCRATCH/analyze.log" 2>&1
# deep-compare metrics.json
if ! diff -u "$HERE/metrics.json" "$SCRATCH/metrics.json"; then
  echo "metrics.json NOT reproducible"; exit 1; fi
echo "metrics.json reproducible"
echo "OK: u5r4 analysis reproducible"