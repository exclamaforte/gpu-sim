#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for i in 01 02 03 04 05 06 07 08; do
  echo "=== window w$i ==="
  python3 measure_window.py --phase 1 --window-id "w$i"
  sleep 12
done
echo "PHASE1 DONE"
