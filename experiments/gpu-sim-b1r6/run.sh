#!/usr/bin/env bash
set -euo pipefail
python scripts/gen_b1r6_metrics.py
python scripts/check_docs_model_numbers.py
python -m unittest tests.test_b1r6_fabric_precision
