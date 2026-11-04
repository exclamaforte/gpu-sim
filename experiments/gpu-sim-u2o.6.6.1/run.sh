#!/usr/bin/env bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# gpu-sim-u2o.6.6.1 -- noise floor of the real measurement path.
# ts queue ONLY (gpu-sim-u2o.6.8 policy). One short job, read-only.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"

echo "=== ts job: $(hostname) at $(date -Is) ==="
python3 measure_noise.py 2>&1 | tee output.log
rc=${PIPESTATUS[0]}
echo "=== measure_noise exit=$rc ==="
exit "$rc"
