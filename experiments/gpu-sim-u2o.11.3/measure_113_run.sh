#!/usr/bin/env bash
# gpu-sim-u2o.11.3 -- the stages 2-5 measure. ts ONLY, one job (measure only;
# binaries prebuilt unqueued by compile_all.py). The script name carries
# "measure_113" so queue_state's own_command_substr identifies this job;
# submit with a description containing "u2o.11.3" for the description half.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
python3 preflight_static.py || { echo "FATAL: gate failed; not submitted"; exit 3; }
echo "=== ts job $(date -Is) trial=${GPQUEUE_TRIAL:-0} ==="
python3 measure_batch.py 2>&1 | tee "${GPQUEUE_TRIAL:+trial_}output.log"
rc=${PIPESTATUS[0]}
echo "=== exit=$rc ==="
exit "$rc"
