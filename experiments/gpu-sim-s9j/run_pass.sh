#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for i in 1 2 3; do
  echo "===== WINDOW $i START $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  systemd-run --user --scope --property=CPUQuota=800% -- \
    python3 measure_session.py --session "$i"
  echo "===== WINDOW $i END $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
done
echo "PASS COMPLETE"
