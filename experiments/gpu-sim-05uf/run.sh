#!/usr/bin/env bash
set -euo pipefail
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-05uf
python scripts/check_docs_model_numbers.py
CI_GATES_FULL=1 python scripts/check_docs_model_numbers.py
python -m pytest tests/test_act_factor_knob.py -q
