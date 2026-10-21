#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Phase 2: bounded, niced load sweep (nice -n 19 burners). Phase label 2.
for b in 1 2 4; do
  for i in 01 02; do
    wid="p2_b${b}_${i}"
    echo "=== window $wid (burners=$b) ==="
    python3 measure_window.py --phase 2 --window-id "$wid" --burners "$b" --load-label "niced-burners=$b"
    sleep 10
  done
done
echo "PHASE2 DONE"
