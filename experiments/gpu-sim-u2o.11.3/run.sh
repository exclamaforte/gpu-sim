#!/usr/bin/env bash
# gpu-sim-u2o.11.3 run.sh: RECOMPUTE from committed files (CPU-only, no GPU).
#
# Re-runs the deterministic tail of the pipeline: merge slices -> verify
# gates -> census negative controls -> unit tests. The forward steps
# (predict keep-list, build manifest, compile unqueued, measure queued) are
# recorded in REPORT.md with job IDs, not recomputed here: compile is
# ~60 CPU-minutes and measurement needs the queued GPU.
#
# Exit nonzero on any gate failure. Byte-identical outputs expected on
# re-run (measured.jsonl, measure_summary.json, metrics.json).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
export CUDA_VISIBLE_DEVICES=
cd "$HERE"

echo "=== 1. census negative controls ==="
python3 census.py

echo "=== 2. unit tests ==="
python3 "$ROOT/tests/test_regime_census.py"
python3 "$ROOT/tests/test_stage_dim_scalars.py"

echo "=== 3. merge slices ==="
python3 merge_slices.py

echo "=== 4. verify gates ==="
python3 verify.py

echo "RECOMPUTE PASS"
