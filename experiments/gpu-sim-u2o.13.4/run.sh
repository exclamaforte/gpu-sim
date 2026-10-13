#!/bin/bash
# gpu-sim-u2o.13.4 replay: table rebuild + accessor tests (CPU-only).
set -u
WT=/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-u2o.13.4
cd "$WT" || exit 1
python3 experiments/gpu-sim-u2o.13.4/build_table_134.py /tmp/134_replay.json || exit 1
python3 - /tmp/134_replay.json experiments/gpu-sim-u2o.13.4/op_latency_table.json <<'EOF' || exit 1
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
assert a["ops"] == b["ops"], "table drift"
print("table rebuilds byte-identical in values")
EOF
CUDA_VISIBLE_DEVICES= python3 -m pytest tests/test_op_latency_134.py -q -p no:cacheprovider || exit 1
echo "REPLAY-OK"
