#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCRATCH="$ROOT/.lab/tmp/elq2"
mkdir -p "$SCRATCH"
python3 experiments/gpu-sim-elq2/gen_metrics.py "$SCRATCH/metrics.scratch.json"
if ! diff -q experiments/gpu-sim-elq2/metrics.json "$SCRATCH/metrics.scratch.json" >/dev/null; then
    echo "MISMATCH: committed metrics.json != freshly computed" >&2
    diff experiments/gpu-sim-elq2/metrics.json "$SCRATCH/metrics.scratch.json" >&2
    exit 1
fi
echo "OK: metrics.json reproducible (12 rows, 5 flip within fabric band)"
