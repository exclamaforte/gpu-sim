#!/usr/bin/env bash
set -euo pipefail
python scripts/gen_b1r6_metrics.py
python scripts/check_gate_citations.py
