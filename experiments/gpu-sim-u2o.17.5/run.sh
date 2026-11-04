#!/usr/bin/env bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# gpu-sim-u2o.17.5 -- capture the canonical real-GPU PyTorch trace fixture.
# ts queue ONLY. No sudo, no clock changes. Fixture only: no pass/fail here.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
echo "=== ts job $(date -Is) ==="
python3 capture_fixture.py 2>&1 | tee output.log
rc=${PIPESTATUS[0]}
echo "=== capture exit=$rc ==="
exit "$rc"
