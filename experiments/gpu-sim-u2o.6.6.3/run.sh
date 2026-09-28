#!/usr/bin/env bash
# gpu-sim-u2o.6.6.3 -- contention characterisation. ts ONLY.
# The burner is started INSIDE this job; no GPU work runs outside the queue.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
# GATE, not a report: two jobs on this branch's predecessors died at import.
GATE="$HERE/preflight_static.py"
python3 "$GATE" || { echo "FATAL: submission gate FAILED"; exit 3; }

echo "=== ts job $(date -Is) ==="
python3 measure_contention.py 2>&1 | tee output.log
rc=${PIPESTATUS[0]}
echo "=== exit=$rc ==="
exit "$rc"
