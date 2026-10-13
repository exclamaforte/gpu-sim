#!/bin/bash
# gpu-sim-u2o.9.1 Stage-2 GPU job: full transformer under 9.5's split.
# Preemptible: SIGUSR1 -> exit 75 (python handler); resume reloads
# ckpt_last.pt and continues (epoch-granular, checkpoint every epoch).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.9.1
DATA=$WT/experiments/gpu-sim-u2o.9.1/train_data_91.jsonl
OUT=/home/gabe/Documents/gpu-sim/runs/gpu-sim-u2o.9.1
cd "$WT" || exit 1
mkdir -p "$OUT"
ARGS="--mode full --data $DATA --out $OUT"
if [ -f "$OUT/ckpt_last.pt" ]; then
  echo "RESUME from ckpt_last.pt"
  ARGS="$ARGS --resume $OUT/ckpt_last.pt"
fi
echo "TRANS-START $(date -u +%FT%TZ)"
# shellcheck disable=SC2086
python3 -m epic_u2o.train_trans_91 $ARGS || exit 1
echo "TRANS-DONE $(date -u +%FT%TZ)"
