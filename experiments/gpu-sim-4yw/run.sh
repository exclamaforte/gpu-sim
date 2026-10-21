#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
REPO="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$REPO/epic_cluster:${PYTHONPATH:-}"
python3 leg1.py > /tmp/scratchpad/4yw_leg1.log 2>&1
grep -q "VERDICT: MODEL-DOES-NOT-REPRODUCE" /tmp/scratchpad/4yw_leg1.log
python3 leg2.py > /tmp/scratchpad/4yw_leg2.log 2>&1
test "$(python3 -c "import json;print(json.load(open('leg2_results.json'))['rows'].__len__())")" = "60"
echo "Leg1 + Leg2 reproduced; verdict MODEL-DOES-NOT-REPRODUCE"
