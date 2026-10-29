#!/usr/bin/env bash
set -euo pipefail
python scripts/gen_7uz2_metrics.py
python scripts/check_docs_model_numbers.py
python -m unittest tests.test_7uz2_act_curve
