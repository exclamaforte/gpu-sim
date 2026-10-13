#!/usr/bin/env bash
# gpu-sim-u2o.6.1.2 -- per-run task containment: CPU enforcement test + GPU proof.
# ts queue ONLY. No sudo, no clock changes, no preflight (see the script header).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
echo "=== ts job $(date -Is) ==="
echo "--- CPU: does the cap actually fire? ---"
python3 "$REPO/tests/test_task_limits.py" 2>&1 | tee cpu_enforcement.log
cpu_rc=${PIPESTATUS[0]}
echo "cpu enforcement exit=$cpu_rc"
echo "--- GPU: real kernel under the cap ---"
python3 verify_capped_run.py 2>&1 | tee output.log
gpu_rc=${PIPESTATUS[0]}
echo "=== cpu_rc=$cpu_rc gpu_rc=$gpu_rc ==="
[ "$cpu_rc" -eq 0 ] && [ "$gpu_rc" -eq 0 ]
