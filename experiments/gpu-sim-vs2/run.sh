#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1. statistic agreement with verdict.py
python3 check_statistic_agreement.py >/tmp/vs2_agree.log 2>&1
echo "statistic agreement: OK"

# 2. recompute each session's spread from its run durations; must match session files
python3 - <<'PY'
import json, glob, os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us
for p in sorted(glob.glob(os.path.join(HERE, "session_*.json"))):
    d=json.load(open(p))
    vals=d["run_durations_us"]
    recomputed=spread_us(vals)
    if abs(recomputed-d["spread_us"])>0.15:
        print("MISMATCH", p, recomputed, d["spread_us"]); raise SystemExit(1)
print("all session spreads recompute OK")
PY

# 3. regenerate metrics.json into scratch and diff against committed
python3 build_metrics.py /tmp/vs2_metrics_regen.json >/tmp/vs2_build.log 2>&1
if ! diff -q /tmp/vs2_metrics_regen.json metrics.json >/dev/null; then
    echo "metrics.json mismatch"; diff /tmp/vs2_metrics_regen.json metrics.json | head; exit 1
fi
echo "metrics.json recompute matches committed: OK"

echo "DONE"
chmod +x run.sh && ./run.sh
