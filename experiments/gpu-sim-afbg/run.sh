#!/usr/bin/env bash
set -euo pipefail
python scripts/gen_afbg_metrics.py
python scripts/check_docs_model_numbers.py
CI_GATES_FULL=1 python scripts/check_docs_model_numbers.py
python -m unittest tests.test_afbg_metrics_top_ranks tests.test_error_budget_ranking
