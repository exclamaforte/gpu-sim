#!/usr/bin/env bash
set -euo pipefail
python scripts/check_docs_model_numbers.py
python -m unittest tests.test_006j_act_shape
