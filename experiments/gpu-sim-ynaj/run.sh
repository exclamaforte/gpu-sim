#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
/home/gabe/pi-research-lab/bin/python3 scripts/effective_cpus.py
/home/gabe/pi-research-lab/bin/python3 -m pytest tests/test_effective_cpus.py tests/test_no_os_cpu_count_in_skips.py -q
echo "OK: effective_cpus helper + guards pass"
