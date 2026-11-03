#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${TMPDIR:-$HERE/../../.lab/tmp}/gpu-sim-wty2-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
U5R4_OUT="$SCRATCH" python3 "$HERE/analyze.py" > "$SCRATCH/analyze.log" 2>&1
echo "compared: $SCRATCH/metrics.json  vs  $HERE/metrics.json"
if ! diff -u "$HERE/metrics.json" "$SCRATCH/metrics.json"; then
  echo "metrics.json NOT reproducible"; exit 1; fi
echo "metrics.json reproducible"
echo "OK: wty2 analysis reproducible"
