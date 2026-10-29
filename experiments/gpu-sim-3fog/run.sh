#!/usr/bin/env bash
set -euo pipefail
python scripts/gen_3fog_metrics.py
python scripts/check_docs_model_numbers.py
