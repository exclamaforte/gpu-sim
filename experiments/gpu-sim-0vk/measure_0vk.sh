#!/usr/bin/env bash
# gpu-sim-0vk -- measure the prebuilt binaries. ts ONLY (measure only;
# binaries prebuilt unqueued by compile_all.py). The script name carries
# "measure_0vk" so queue_state's own_command_substr identifies this job;
# submit with a description containing "gpu-sim-0vk" for the description half.
# GPQUEUE_TRIAL=1: trial leg (first 30 specs, binaries kept).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
echo "=== ts job $(date -Is) trial=${GPQUEUE_TRIAL:-0} ==="
python3 measure_0vk.py 2>&1 | tee "${GPQUEUE_TRIAL:+trial_}output.log"
rc=${PIPESTATUS[0]}
echo "=== exit=$rc ==="
exit "$rc"
