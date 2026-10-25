#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python experiments/gpu-sim-bw5/phase_map.py >/tmp/bw5.out
python3 -c "import json; d=json.load(open('experiments/gpu-sim-bw5/metrics.json')); assert len(d['records'])==24"
echo "OK"
