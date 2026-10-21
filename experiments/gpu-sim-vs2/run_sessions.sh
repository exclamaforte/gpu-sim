#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
run_s1() { python3 measure_session.py --setting S1 --session "$1"; }
run_s2() { systemd-run --user --scope --property=CPUQuota=800% -- \
            python3 measure_session.py --setting S2 --session "$1"; }
for i in 1 2 3; do
  echo "===== S1 session $i ====="
  run_s1 "$i"
  sleep 8
  echo "===== S2 session $i ====="
  run_s2 "$i"
  sleep 8
done
echo "ALL SESSIONS DONE"
