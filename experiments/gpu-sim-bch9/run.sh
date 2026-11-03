#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SCRATCH="$ROOT/.lab/tmp/bch9"
mkdir -p "$SCRATCH"
python3 experiments/gpu-sim-bch9/gen_metrics.py "$SCRATCH/metrics.scratch.json"
if ! diff -q experiments/gpu-sim-bch9/metrics.json "$SCRATCH/metrics.scratch.json" >/dev/null; then
    echo "MISMATCH: committed metrics.json != freshly computed" >&2
    diff experiments/gpu-sim-bch9/metrics.json "$SCRATCH/metrics.scratch.json" >&2
    exit 1
fi
python3 - << 'PY'
import json
d = json.load(open("experiments/gpu-sim-bch9/metrics.json"))
a = d["aggregates"]
print("OK: metrics.json reproducible (rows=%d, flip46=%d, invariant=%d, worst_regret=%.4f%%, n_factors=%d)" % (
    len(d["rows"]), a["n_rows_flipping_46"], a["n_invariant"], a["worst_regret_pct_max"], d["grid"]["n_total"]))
PY
