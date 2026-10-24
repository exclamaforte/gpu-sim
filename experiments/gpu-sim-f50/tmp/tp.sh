#!/usr/bin/env bash
# f50 TP gate: sharded fwd == reference, 20 SGD steps match + fall.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-f50
export PYTHONPATH=$PWD
mkdir -p experiments/gpu-sim-f50/tmp
export TMPDIR=$PWD/experiments/gpu-sim-f50/tmp
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NCCL_NO_FILL=1 FAKEGPU_NUM_DEVICES=2
export GPUSIM_TOPOLOGY=$PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin
python3 -m tools.torch_parallel.topology --ranks 2 --fabric nvlink \
    --out $PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin > /dev/null
export FAKEGPU_NCCL_SIMULATE_ONLY=0
rm -f /dev/shm/fgcoll_*
GPUSIM_TRACE=$PWD/experiments/gpu-sim-f50/tmp/f50-tp.jsonl \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    experiments/gpu-sim-f50/tp_gate.py
tprc=$?
echo "tp_gate rc=$tprc"
test $tprc -eq 0 || exit 1
