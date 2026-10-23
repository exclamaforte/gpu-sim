#!/usr/bin/env bash
# tbp equivalence: world1 vs world2 --dp-mean, SGD, same seed/batch.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-tbp
mkdir -p experiments/gpu-sim-tbp/tmp
export TMPDIR=$PWD/experiments/gpu-sim-tbp/tmp
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NCCL_NO_FILL=1
export FAKEGPU_NCCL_SIMULATE_ONLY=0
export GPUSIM_TOPOLOGY=$PWD/experiments/gpu-sim-tbp/tmp/tbp-topo.bin
python3 -m tools.torch_parallel.topology --ranks 2 --fabric nvlink \
    --out $PWD/experiments/gpu-sim-tbp/tmp/tbp-topo.bin > /dev/null
T=$PWD/experiments/gpu-sim-tbp/tmp
rm -f /dev/shm/fgcoll_*  # crash leftovers from earlier runs
rm -f $T/tbp-w1.json $T/tbp-w2.json  # train.py APPENDS per-rank rows
echo "--- world1 baseline (12 steps, sgd lr 0.01) ---"
FAKEGPU_NUM_DEVICES=1 GPUSIM_TRACE=$T/tbp-w1.jsonl \
python3 -m torch.distributed.run --nproc_per_node=1 --standalone \
    tools/torch_parallel/train.py --steps 12 --warmup 2 --optim sgd --lr 0.01 \
    --seed 7 --json $T/tbp-w1.json
echo "--- world2 --dp-mean (12 steps, same seed) ---"
FAKEGPU_NUM_DEVICES=2 GPUSIM_TRACE=$T/tbp-w2.jsonl \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    tools/torch_parallel/train.py --steps 12 --warmup 2 --optim sgd --lr 0.01 \
    --seed 7 --dp-mean --json $T/tbp-w2.json
python3 experiments/gpu-sim-tbp/tbp_check.py $T/tbp-w1.json $T/tbp-w2.json
