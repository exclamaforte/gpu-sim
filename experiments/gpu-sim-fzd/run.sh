#!/usr/bin/env bash
set -euo pipefail
cd /home/gabe/Documents/gpu-sim
export LD_PRELOAD=$PWD/libfakenccl.so
export FAKEGPU_NUM_DEVICES=1
export GPUSIM_TOPOLOGY=/tmp/fzd_topo.bin
python3 -m tools.torch_parallel.topology --ranks 2 --fabric nvlink --out /tmp/fzd_topo.bin > /dev/null 2>&1
echo "=== allreduce (crude) ==="
FAKEGPU_NCCL_SIMULATE_ONLY=0 python3 -m torch.distributed.run --nproc_per_node=2 --standalone tools/torch_parallel/train.py --steps 5 --warmup 1 --cp 2 --seq 64 --d-model 32 --layers 2 --vocab 256 --seed 42 --cp-grad-sync allreduce 2>&1 | grep "loss\|mean step" | head -n 4
echo "=== reducescatter (production) ==="
FAKEGPU_NCCL_SIMULATE_ONLY=0 python3 -m torch.distributed.run --nproc_per_node=2 --standalone tools/torch_parallel/train.py --steps 5 --warmup 1 --cp 2 --seq 64 --d-model 32 --layers 2 --vocab 256 --seed 42 --cp-grad-sync reducescatter 2>&1 | grep "loss\|mean step" | head -n 4
