#!/usr/bin/env bash
set -euo pipefail
python scripts/error_budget.py --json >/dev/null
python scripts/check_docs_model_numbers.py
CI_GATES_FULL=1 python scripts/check_docs_model_numbers.py
python -m unittest tests.test_error_budget_ranking
