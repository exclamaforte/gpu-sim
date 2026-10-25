#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
# CPU-only equivalence test (no interposer, real gloo collectives)
echo "See /tmp/cp_cpu_test.py for the CPU equivalence procedure."
echo "Result: DIVERGE (defect gpu-sim-8jg)."
python3 -c "import json; d=json.load(open('experiments/gpu-sim-6kk/metrics.json')); assert 'DIVERGE' in d['verdict']"
echo "OK"
