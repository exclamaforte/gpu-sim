#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -c "import json; d=json.load(open('experiments/gpu-sim-8jg/metrics.json')); assert d['max_abs_param_diff_cp2_rank0_vs_rank1']==0.0 or d['max_abs_param_diff_cp2_rank0_vs_rank1']<1e-6"
echo "OK"
