#!/usr/bin/env bash
# gpu-sim-7q3w.4: one invocation of each deliverable increment.
#
#   1  probe_concurrent.py  8 concurrent node processes, each sees 8 B300s,
#                           with the RAM preflight and its refusal arm
#   2  probe_distributed.py torch.distributed world_size 8 (gloo), one rank
#                           per node; per-rank bus ids and all_reduce agreement
#   3  probe_training.py    12.6M transformer + AdamW, 3 real DDP steps,
#                           canonical checkpoint hash equal on all 8 ranks
#
# `run.sh 1|2|3` runs one increment; no argument runs all three in order.  The
# probes build their own shims from specs/b300_8x8.json.  The model math is
# CPU float32 + gloo; CUDA is visible so torch's NVML/driver identity and
# AdamW's accelerator health check work, but no GPU kernel/training claim is
# made here.  capture.py runs each increment separately, waits for that
# increment's own preflight floor, and writes output.log, metrics.json and
# job.json from the actual runs.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXP="$PWD/experiments/gpu-sim-7q3w.4"
ONLY="${1:-all}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_one() {
    case "$1" in
        0)
            echo "=== increment 0: A2 node isolation + its negative control ==="
            python3 "$EXP/probe_a2.py"
            ;;
        1)
            echo "=== increment 1: eight concurrent node processes + RAM preflight ==="
            python3 "$EXP/probe_concurrent.py"
            ;;
        2)
            echo "=== increment 2: torch.distributed world_size 8, one rank per node ==="
            python3 "$EXP/probe_distributed.py"
            ;;
        3)
            echo "=== increment 3: 12.6M transformer + AdamW, 8 DDP ranks, equal checkpoints ==="
            python3 "$EXP/probe_training.py"
            ;;
        4)
            echo "=== increment 4: A5 8-way DDP vs single-rank reference ==="
            python3 "$EXP/probe_reference.py"
            ;;
        5)
            echo "=== REFUSE control: 10496 < 10576 ==="
            python3 "$EXP/probe_training.py" --force-refuse-mib 10496
            ;;
        *)
            echo "usage: $0 [0|1|2|3|4|5]" >&2
            exit 2
            ;;
    esac
}

if [ "$ONLY" = "all" ]; then
    run_one 0
    echo
    run_one 1
    echo
    run_one 2
    echo
    run_one 3
    echo
    run_one 4
else
    run_one "$ONLY"
fi
