#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ ! -f specs/h100_rack_4node.json ]; then
    python experiments/gpu-sim-bw5/gen_spec.py
fi
python experiments/gpu-sim-bw5/phase_map.py >/tmp/bw5.out
python3 -c "import json; d=json.load(open('experiments/gpu-sim-bw5/metrics.json')); assert len(d['records'])==24"
echo "OK"
