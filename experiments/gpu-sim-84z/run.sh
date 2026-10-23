#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${TMPDIR:-/tmp}/gpu-sim-84z-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$HERE/metrics.json" "$SCRATCH/metrics.before.json"
python3 "$HERE/compute_84z.py" >/dev/null
diff -u "$SCRATCH/metrics.before.json" "$HERE/metrics.json" && echo "metrics.json reproducible"
