#!/usr/bin/env bash
# f50 default-gate replay: SIMULATE_ONLY unset -> world-4 DP traffic must
# reproduce the committed 8y7.3 record (149 coll / 86.1 MB). train.py/model.py
# changed under this bead, so the default gate must be re-proven.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-f50
mkdir -p experiments/gpu-sim-f50/tmp
export TMPDIR=$PWD/experiments/gpu-sim-f50/tmp
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NCCL_NO_FILL=1
export GPUSIM_TOPOLOGY=$PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin
python3 -m tools.torch_parallel.topology --ranks 4 --fabric nvlink \
    --out $PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin > /dev/null
export FAKEGPU_NUM_DEVICES=4
unset FAKEGPU_NCCL_SIMULATE_ONLY
rm -f /dev/shm/fgcoll_*
python3 -m tools.torch_parallel.sweep --world-sizes 4 --steps 5 \
    --fabric nvlink > $PWD/experiments/gpu-sim-f50/tmp/f50-gate.txt 2>&1
tail -2 $PWD/experiments/gpu-sim-f50/tmp/f50-gate.txt
python3 experiments/gpu-sim-n5h/gate_check.py \
    $PWD/experiments/gpu-sim-f50/tmp/f50-gate.txt || exit 1
