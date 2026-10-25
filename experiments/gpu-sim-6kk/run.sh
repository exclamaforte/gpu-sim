#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# CPU-only equivalence test (no interposer, real gloo collectives).
# cp=1 single-process smoke; cp=2 requires 2 processes (see cp_cpu_equiv.py).
WORLD_SIZE=1 RANK=0 python3 experiments/gpu-sim-6kk/cp_cpu_equiv.py >/tmp/6kk_smoke.log 2>&1
python3 -c "import json; d=json.load(open('experiments/gpu-sim-6kk/metrics.json')); assert 'DIVERGE' in d['verdict']"
echo "OK"
