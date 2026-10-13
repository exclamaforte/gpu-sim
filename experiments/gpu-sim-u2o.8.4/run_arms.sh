#!/bin/bash
# gpu-sim-u2o.8.4: all three ablation arms in ONE ts job (design: budget
# bounded, arms sequential). Progress tokens: ARM-START/ARM-DONE per arm;
# TrainingLoop logs every ~10 steps + val MAPE each half-epoch.
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.8.4
DATA=$WT/experiments/gpu-sim-u2o.8.4/ablation_data.jsonl
OUT=/home/gabe/Documents/gpu-sim/runs/gpu-sim-u2o.8.4
cd "$WT" || exit 1
mkdir -p "$OUT"
for ARM in source ptx hybrid; do
  echo "ARM-START $ARM $(date -u +%FT%TZ)"
  python3 -m epic_u2o.train_ablation --arm "$ARM" --data "$DATA" --out "$OUT/$ARM" || exit 1
  echo "ARM-DONE $ARM $(date -u +%FT%TZ)"
done
python3 "$WT/experiments/gpu-sim-u2o.8.4/compare_84.py" "$OUT"
echo "ABLATION-COMPLETE $(date -u +%FT%TZ)"
