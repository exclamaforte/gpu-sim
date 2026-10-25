#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m pytest tests/test_attention_flops.py -q
echo "OK"
