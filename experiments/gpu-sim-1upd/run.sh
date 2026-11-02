#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
env PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/ -q 2>&1 | tail -3
echo "bytecode check: no __pycache__ under experiments/gpu-sim-1upd/"
test -z "$(find experiments/gpu-sim-1upd -name '__pycache__' -type d 2>/dev/null)" && echo "OK: no pycache in experiments/gpu-sim-1upd/" || { echo "FAIL: pycache found"; exit 1; }
