#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
test -f experiments/gpu-sim-oez/decision.md
test -f experiments/gpu-sim-oez/metrics.json
python3 -c "import json; d=json.load(open('experiments/gpu-sim-oez/metrics.json')); assert d['decision']=='b'"
echo "OK"
