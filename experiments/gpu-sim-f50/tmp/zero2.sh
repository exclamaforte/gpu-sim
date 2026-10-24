#!/usr/bin/env bash
# f50 ZeRO-2 gate: world1 saves flat grads; world2 zero2 reassembles + verifies.
set -u
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-f50
mkdir -p experiments/gpu-sim-f50/tmp
export TMPDIR=$PWD/experiments/gpu-sim-f50/tmp
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NCCL_NO_FILL=1
export FAKEGPU_NCCL_SIMULATE_ONLY=0
export GPUSIM_TOPOLOGY=$PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin
python3 -m tools.torch_parallel.topology --ranks 2 --fabric nvlink \
    --out $PWD/experiments/gpu-sim-f50/tmp/f50-topo.bin > /dev/null
T=$PWD/experiments/gpu-sim-f50/tmp
rm -f /dev/shm/fgcoll_* $T/f50-ref-flat.pt
echo "--- world1 save flat grads (3 steps) ---"
FAKEGPU_NUM_DEVICES=1 \
python3 -m torch.distributed.run --nproc_per_node=1 --standalone \
    tools/torch_parallel/train.py --steps 3 --warmup 1 --optim sgd --lr 0.01 \
    --seed 7 --save-flat $T/f50-ref-flat.pt --json $T/f50-z2-w1.json
echo "--- world2 zero2 verify (3 steps) ---"
FAKEGPU_NUM_DEVICES=2 \
python3 -m torch.distributed.run --nproc_per_node=2 --standalone \
    tools/torch_parallel/train.py --steps 3 --warmup 1 --optim sgd --lr 0.01 \
    --seed 7 --zero 2 --dp-mean --verify-flat $T/f50-ref-flat.pt \
    --json $T/f50-z2-w2.json
z2rc=$?
echo "zero2 rc=$z2rc (asserts fire in-train per step)"
test $z2rc -eq 0 || exit 1
