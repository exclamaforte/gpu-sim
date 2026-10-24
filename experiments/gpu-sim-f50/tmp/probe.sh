#!/usr/bin/env bash
# f50 probe job: all_gather, reduce_scatter, send/recv vs host references.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-f50
mkdir -p experiments/gpu-sim-f50/tmp
export TMPDIR=$PWD/experiments/gpu-sim-f50/tmp
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NCCL_NO_FILL=1 FAKEGPU_NUM_DEVICES=2
export GPUSIM_TOPOLOGY=$PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin
python3 -m tools.torch_parallel.topology --ranks 2 --fabric nvlink \
    --out $PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin > /dev/null
export FAKEGPU_NCCL_SIMULATE_ONLY=0
rm -f /dev/shm/fgcoll_*
GPUSIM_TRACE=$PWD/experiments/gpu-sim-f50/tmp/f50-probe.jsonl \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    experiments/gpu-sim-f50/ag_rs_p2p_probe.py
echo "probe rc=$?"
