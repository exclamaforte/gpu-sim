#!/usr/bin/env bash
# gpu-sim-8y7.3 acceptance: fixed world 4, ZeRO 1 vs 2 vs 3 (dp=4).
# GPU job body -- queue with:
#   ts -L "gpu-sim-8y7.3: acceptance" --max-runtime 10m experiments/gpu-sim-8y7.3/accept.sh
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
run zero1 --world-sizes 4 --zero 1
run zero2 --world-sizes 4 --zero 2
run zero3 --world-sizes 4 --zero 3
echo "acceptance done" | tee -a "$OUT/output.log"
