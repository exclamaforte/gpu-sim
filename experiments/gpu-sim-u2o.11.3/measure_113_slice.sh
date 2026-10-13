#!/usr/bin/env bash
# u2o.11.3 slice job: measure ONE manifest slice. ts ONLY.
# No preflight inside the GPU slot: it passed on this exact tree+binaries
# (locally and in the trial leg), and 32 more nvcc runs per slice job would
# idle the device the queue entrusted to measurement. The name carries
# "measure_113" for queue_state's own_command_substr; submit with a
# description containing "u2o.11.3 <tag>" for the description half.
# Env: U113_SLICE="lo:hi" (required), U113_TAG (required).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$HERE"
: "${U113_SLICE:?slice lo:hi required}"
: "${U113_TAG:?tag required}"
echo "=== ts slice ${U113_TAG} [${U113_SLICE}] $(date -Is) ==="
python3 measure_batch.py 2>&1 | tee "slice_${U113_TAG}.log"
rc=${PIPESTATUS[0]}
echo "=== exit=$rc ==="
exit "$rc"
