#!/usr/bin/env bash
# gpu-sim-e5q -- the large-size measure. ts ONLY, one job (measure only;
# binaries prebuilt unqueued by compile_all.py).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
python3 preflight_static.py || { echo "FATAL: gate failed; not submitted"; exit 3; }
echo "=== ts job $(date -Is) ==="
python3 measure_batch.py 2>&1 | tee output.log
rc=${PIPESTATUS[0]}
echo "=== exit=$rc ==="
exit "$rc"
