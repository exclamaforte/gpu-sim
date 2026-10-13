#!/bin/bash
# gpu-sim-u2o.9.2 fusion comparison: concat vs cross_attention, one GPU job.
# Matched: same 39M-class encoder, hparams, split (9.5 family_holdout_v1),
# seed -- only fusion_type differs (cross_attn carries +1.3% params from its
# QKVO projections; counts recorded, mechanism comparison stated as such).
# Short job (~3 min observed class): --max-runtime 10m smoke path, no trial.
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.9.2
DATA=$WT/experiments/gpu-sim-u2o.9.1/train_data_91.jsonl
OUT=/home/gabe/Documents/gpu-sim/runs/gpu-sim-u2o.9.2
cd "$WT" || exit 1
mkdir -p "$OUT"
for F in concat cross_attention; do
  echo "FUSION-START $F $(date -u +%FT%TZ)"
  python3 -m epic_u2o.train_trans_91 --mode full --data "$DATA" \
      --out "$OUT/$F" --fusion-type "$F" || exit 1
  echo "FUSION-DONE $F $(date -u +%FT%TZ)"
done
python3 "$WT/experiments/gpu-sim-u2o.9.2/compare_92.py" "$OUT"
echo "FUSION-COMPLETE $(date -u +%FT%TZ)"
