#!/usr/bin/env bash
# gpu-sim-7o2 main: 7 interleaved rounds (28 reps) + probe round = 8 per cell.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-7o2
mkdir -p experiments/gpu-sim-7o2/tmp
export TMPDIR=$PWD/experiments/gpu-sim-7o2/tmp
unset FAKEGPU_NCCL_SIMULATE_ONLY
rm -f /dev/shm/fgcoll_*
python3 experiments/gpu-sim-7o2/run_interleaved.py --rounds 7 --seed 702 \
    --round-offset 1 --out $PWD/experiments/gpu-sim-7o2/reps_main.jsonl
rc=$?; echo "main rc=$rc"; test $rc -eq 0
