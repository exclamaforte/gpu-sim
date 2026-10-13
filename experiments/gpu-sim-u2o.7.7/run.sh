#!/bin/bash
# gpu-sim-u2o.7.7 merged-corpus audit replay (CPU only, no GPU, no ts job).
# Re-runs the merged audit + the split test module; outputs must be
# byte-identical to the committed metrics.json (modulo key order, which
# json.dump sorts) and 18/18 tests green.
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.7.7
EXP=$WT/experiments/gpu-sim-u2o.7.7
cd "$WT" || exit 1
cp "$EXP/metrics.json" /tmp/77_metrics_before.json
CUDA_VISIBLE_DEVICES= python3 "$EXP/run_merged_audit.py" || exit 1
if ! python3 -c "import json; a=json.load(open('/tmp/77_metrics_before.json')); b=json.load(open('$EXP/metrics.json')); assert a==b, 'metrics drift on replay'"; then
  echo "REPLAY-DRIFT"; exit 1
fi
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_dataset_split.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
