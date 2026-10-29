#!/usr/bin/env bash
set -euo pipefail
python scripts/check_docs_model_numbers.py
CI_GATES_FULL=1 python scripts/check_docs_model_numbers.py
python -m pytest tests/test_act_factor_knob.py -q
