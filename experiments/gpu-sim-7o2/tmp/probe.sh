#!/usr/bin/env bash
# gpu-sim-7o2 probe: one interleaved round (4 reps) to calibrate per-rep wall time.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-7o2
mkdir -p experiments/gpu-sim-7o2/tmp
export TMPDIR=$PWD/experiments/gpu-sim-7o2/tmp
unset FAKEGPU_NCCL_SIMULATE_ONLY
rm -f /dev/shm/fgcoll_*
python3 experiments/gpu-sim-7o2/run_interleaved.py --rounds 1 --seed 702 \
    --round-offset 0 --out $PWD/experiments/gpu-sim-7o2/reps_probe.jsonl
echo "probe rc=$?"
