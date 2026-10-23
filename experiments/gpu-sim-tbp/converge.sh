#!/usr/bin/env bash
# tbp convergence: world2 --dp-mean, 50 steps SGD; the curve must really fall.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-tbp
mkdir -p experiments/gpu-sim-tbp/tmp
export TMPDIR=$PWD/experiments/gpu-sim-tbp/tmp
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NCCL_NO_FILL=1
export FAKEGPU_NCCL_SIMULATE_ONLY=0
export FAKEGPU_NUM_DEVICES=2
export GPUSIM_TOPOLOGY=$PWD/experiments/gpu-sim-tbp/tmp/tbp-topo.bin
python3 -m tools.torch_parallel.topology --ranks 2 --fabric nvlink \
    --out $PWD/experiments/gpu-sim-tbp/tmp/tbp-topo.bin > /dev/null
T=$PWD/experiments/gpu-sim-tbp/tmp
rm -f /dev/shm/fgcoll_* $T/tbp-w2long.json
echo "--- world2 --dp-mean 50 steps, sgd lr 0.01 ---"
START=$SECONDS
FAKEGPU_NUM_DEVICES=2 GPUSIM_TRACE=$T/tbp-w2long.jsonl \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    tools/torch_parallel/train.py --steps 50 --warmup 2 --optim sgd --lr 0.01 \
    --seed 7 --dp-mean --json $T/tbp-w2long.json
echo "wall $((SECONDS - START))s for 50 steps"
python3 experiments/gpu-sim-tbp/tbp_check.py $T/tbp-w1.json $T/tbp-w2.json \
    --long $T/tbp-w2long.json
