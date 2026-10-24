#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -c "import json; d=json.load(open('experiments/gpu-sim-5o4.6/metrics.json')); assert 'N8' in d and 'N16' in d"
python -m pytest tests/test_topology_search.py -q
echo "OK"
