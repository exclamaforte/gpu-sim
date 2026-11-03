#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/home/gabe/pi-research-lab/bin/python3 scripts/check_append_only.py
echo "append-only + ledger test-id resolution OK"
