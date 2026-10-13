#!/bin/bash
# gpu-sim-u2o.9.1 CPU replay (no GPU, no ts job): data build + negctrl +
# tests, byte-identical where deterministic. The GPU train (run_trans.sh,
# ~40 s, 157 MB ckpt NOT committed) is documented, not replayed.
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.9.1
EXP=$WT/experiments/gpu-sim-u2o.9.1
cd "$WT" || exit 1
cp "$EXP/train_data_91.jsonl" /tmp/91_data_before.jsonl
cp "$EXP/negctrl.json" /tmp/91_neg_before.json
CUDA_VISIBLE_DEVICES= python3 "$EXP/build_data_91.py" || exit 1
python3 -c "import json; a=[json.loads(l) for l in open('/tmp/91_data_before.jsonl')]; b=[json.loads(l) for l in open('$EXP/train_data_91.jsonl')]; assert a==b, 'data drift'" || { echo REPLAY-DRIFT; exit 1; }
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$EXP/negctrl_91.py" || exit 1
python3 -c "import json; a=json.load(open('/tmp/91_neg_before.json')); b=json.load(open('$EXP/negctrl.json')); assert a==b, 'negctrl drift'" || { echo REPLAY-DRIFT; exit 1; }
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_trans_baseline.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
