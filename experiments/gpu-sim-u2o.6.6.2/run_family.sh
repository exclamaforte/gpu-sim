#!/usr/bin/env bash
# gpu-sim-u2o.6.6.2 -- one anchor family per ts job.
# usage: run_family.sh <launch|memory|reduce|gemm>
# ts queue ONLY; no sudo, no clock changes, no untrusted kernels, no preflight.
set -uo pipefail
FAM="${1:?usage: run_family.sh <family>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"

echo "=== ts job $(date -Is) family=$FAM ==="
python3 measure_family.py "$FAM" 2>&1 | tee "output_${FAM}.log"
rc=${PIPESTATUS[0]}
echo "=== family=$FAM exit=$rc ==="
exit "$rc"
