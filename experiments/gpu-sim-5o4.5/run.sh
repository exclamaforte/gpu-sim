#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python experiments/gpu-sim-5o4.5/sensitivity.py >/tmp/5o4.5.out
python -c "import json; d=json.load(open('experiments/gpu-sim-5o4.5/metrics.json')); assert d['verdict']=='not separable at this margin'"
echo "OK"
