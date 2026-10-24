#!/usr/bin/env bash
set -euo pipefail
REPO=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-5o4.7
OUT=$REPO/experiments/gpu-sim-5o4.7/raw
mkdir -p "$OUT"
export LD_PRELOAD=$REPO/libfakenccl.so
export GPUSIM_TRACE=$OUT/trace.jsonl
export FAKEGPU_NCCL_NO_FILL=1
export FAKEGPU_NUM_DEVICES=8
rm -f $OUT/trace.jsonl $OUT/summary.jsonl
cd "$REPO"
python -m torch.distributed.run --nproc_per_node=8 --standalone \
  tools/torch_parallel/train.py \
  --steps 5 --warmup 2 --batch 8 \
  --tp 2 --pp 1 --cp 4 --zero 1 \
  --seed 0 --vocab 4096 --seq 512 --d-model 256 --layers 4 \
  --json $OUT/summary.jsonl
echo "DONE"
