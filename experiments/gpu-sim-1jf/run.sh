#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${TMPDIR:-/tmp}/gpu-sim-1jf-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
python3 "$HERE/compute_metrics.py" --out "$SCRATCH/metrics.json"
diff -u "$HERE/metrics.json" "$SCRATCH/metrics.json" && echo "metrics.json matches recomputed output"
