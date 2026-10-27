#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/microbatch_sweep.py >/dev/null
python3 -m pytest tests/test_microbatch_sweep.py -q 2>&1 | tail -1
