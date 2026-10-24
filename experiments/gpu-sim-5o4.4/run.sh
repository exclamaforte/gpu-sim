#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m epic_cluster.parallelism_search > /tmp/5o4.4.out
grep -q "WINNER: tp=1 pp=1 cp=1 dp=8 zero=3" /tmp/5o4.4.out
echo "OK: winner confirmed"
