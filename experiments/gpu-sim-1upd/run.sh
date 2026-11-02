#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
env PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/ -q 2>&1 | tail -3
echo "bytecode check: no __pycache__ under experiments/"
test -z "$(find experiments -name '__pycache__' -type d 2>/dev/null)" && echo "OK: no pycache in experiments/" || { echo "FAIL: pycache found"; exit 1; }
