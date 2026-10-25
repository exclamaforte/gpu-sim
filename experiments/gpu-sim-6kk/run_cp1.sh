#!/usr/bin/env bash
set -euo pipefail
REPO=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-6kk
OUT=$REPO/experiments/gpu-sim-6kk/raw_cp1
mkdir -p "$OUT"
export LD_PRELOAD=$REPO/libfakenccl.so
export GPUSIM_TRACE=$OUT/trace.jsonl
export FAKEGPU_NUM_DEVICES=1
rm -f $OUT/trace.jsonl $OUT/summary.jsonl
cd "$REPO"
python -m torch.distributed.run --nproc_per_node=1 --standalone \
  tools/torch_parallel/train.py \
  --steps 8 --warmup 2 --batch 4 \
  --tp 1 --pp 1 --cp 1 --zero 1 \
  --seed 42 --vocab 1024 --seq 128 --d-model 64 --layers 2 \
  --json $OUT/summary.jsonl
echo "DONE cp1"
