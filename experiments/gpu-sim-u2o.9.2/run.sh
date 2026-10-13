#!/bin/bash
# gpu-sim-u2o.9.2 CPU replay (no GPU): compare + fusion tests.
# The GPU arms (run_fusion.sh, ~40 s/arm) are documented, not replayed.
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.9.2
EXP=$WT/experiments/gpu-sim-u2o.9.2
cd "$WT" || exit 1
cp "$EXP/metrics.json" /tmp/92_metrics_before.json
python3 "$EXP/compare_92.py" /home/gabe/Documents/gpu-sim/runs/gpu-sim-u2o.9.2 || exit 1
python3 -c "import json; a=json.load(open('/tmp/92_metrics_before.json')); b=json.load(open('$EXP/metrics.json')); assert a==b, 'metrics drift'" || { echo REPLAY-DRIFT; exit 1; }
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_fusion.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
