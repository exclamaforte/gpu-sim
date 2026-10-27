#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 scripts/gen_claims_register.py --check
python3 -m pytest tests/test_b5f2_radius_prose.py -q
