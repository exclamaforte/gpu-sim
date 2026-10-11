#!/usr/bin/env bash
# gpu-sim-f6y -- measure the 80 fixed-template binaries. ts ONLY, one job
# (measure only; binaries prebuilt unqueued by compile_all.py). The script
# name carries "measure_f6y" so queue_state's own_command_substr identifies
# this job; submit with a description containing "gpu-sim-f6y" for the
# description half.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
echo "=== ts job $(date -Is) ==="
python3 measure_f6y.py 2>&1 | tee "output.log"
rc=${PIPESTATUS[0]}
echo "=== exit=$rc ==="
exit "$rc"
