#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m pytest tests/test_close_audit_unchanged.py -q
