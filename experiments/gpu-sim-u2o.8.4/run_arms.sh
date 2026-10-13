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
# Preemptible (arm-granular): SIGUSR1 -> exit 75 for requeue. A finished arm
# owns arm_metrics.json and is skipped on resume; a partial arm owns none
# and restarts clean (entrypoint wipes ckpt/logs at start). 11 lines.
trap 'echo "PREEMPTED"; exit 75' USR1
MODEL_ARGS="${MODEL_ARGS:-}"  # e.g. "--d-model 256 --n-heads 8 --n-layers 6 --d-ff 1024"
IDX=0
for ARM in source ptx hybrid; do
  if [ -f "$OUT/$ARM/arm_metrics.json" ]; then
    echo "ARM-SKIP $ARM (resume: complete)"
  else
    echo "ARM-START $ARM $(date -u +%FT%TZ)"
    # shellcheck disable=SC2086
    python3 -m epic_u2o.train_ablation --arm "$ARM" --arm-index "$IDX" --n-arms 3 $MODEL_ARGS --data "$DATA" --out "$OUT/$ARM" || exit 1
    echo "ARM-DONE $ARM $(date -u +%FT%TZ)"
  fi
  IDX=$((IDX + 1))
done
python3 "$WT/experiments/gpu-sim-u2o.8.4/compare_84.py" "$OUT"
echo "ABLATION-COMPLETE $(date -u +%FT%TZ)"
