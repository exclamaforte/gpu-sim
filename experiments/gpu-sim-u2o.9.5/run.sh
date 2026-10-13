#!/bin/bash
# gpu-sim-u2o.9.5 replay (CPU only, no GPU, no ts job: HGBR on 456x33).
# Re-runs the baseline + tests; metrics.json must be byte-identical.
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.9.5
EXP=$WT/experiments/gpu-sim-u2o.9.5
cd "$WT" || exit 1
cp "$EXP/metrics.json" /tmp/95_metrics_before.json
CUDA_VISIBLE_DEVICES= python3 "$EXP/train_gbt_95.py" || exit 1
python3 -c "import json; a=json.load(open('/tmp/95_metrics_before.json')); b=json.load(open('$EXP/metrics.json')); assert a==b, 'metrics drift on replay'" || { echo REPLAY-DRIFT; exit 1; }
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_gbt_baseline.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
