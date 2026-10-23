#!/usr/bin/env bash
# gpu-sim-8y7.1 acceptance: DP=4 vs TP=4 vs PP=2 (dp=2) vs PP=4 (dp=1),
# same model, world 4, nvlink. GPU job body -- queue with:
#   ts -L "gpu-sim-8y7.1: acceptance" --max-runtime 60m experiments/gpu-sim-8y7.1/accept.sh
# Writes sweep JSONs + console output into experiments/gpu-sim-8y7.1/.
# Uses only the worktree it lives in; safe to re-run (overwrites its own outputs).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."  # worktree root
export GPUSIM_ALLOW_EXTRAPOLATE=1
OUT="$HERE"
: > "$OUT/output.log"
run() {
    local name="$1"; shift
    echo "=== $name ===" | tee -a "$OUT/output.log"
    # shellcheck disable=SC2068
    python3 -m tools.torch_parallel.sweep $@ --steps 5 --fabric nvlink \
        --json "$OUT/sweep-$name.json" 2>&1 | tee -a "$OUT/output.log"
    echo "GPQUEUE_PROGRESS sweep-$name done rc=${PIPESTATUS[0]}"
}
run dp4   --world-sizes 4
run tp4   --world-sizes 4 --tp 4
run pp2   --world-sizes 4 --pp 2 --micro-batches 2
run pp4   --world-sizes 4 --pp 4 --micro-batches 2
echo "acceptance done" | tee -a "$OUT/output.log"
