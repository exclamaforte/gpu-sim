#!/bin/bash
# CPU-only: recompute the K-window summary from the committed rows and require it to
# match the committed metrics.json. The measurement itself ran on the GPU as job
# 20261105t045608-9367da1b (see job.json); this recipe must never need a GPU, or the
# gate (CPU-side) cannot show the artifact is reproducible.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$HERE/summarize.py" --rows "$HERE/rows.jsonl" --out "$HERE/.recomputed_metrics.json" --compare "$HERE/metrics.json"
echo "METRICS_MATCH"
