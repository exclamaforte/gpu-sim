#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/home/gabe/pi-research-lab/bin/python3 -m pytest tests/test_no_unconditional_skips.py -q
echo "guard passed: no unconditional skips"
