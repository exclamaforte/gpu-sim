#!/usr/bin/env bash
# gpu-sim-u2o.6.1.1 -- bisect the run_kernel real-GPU failure, then re-run the
# full measure_spec pipeline with the fix applied. ts-queue only.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
python3 bisect_run_kernel.py 2>&1 | tee bisect.log
rc=${PIPESTATUS[0]}
echo "=== bisect exit=$rc ==="
exit "$rc"
