#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

python3 check_statistic_agreement.py >/dev/null 2>&1
echo "statistic agreement: OK"

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

python3 build_metrics.py /home/gabe/Documents/gpu-sim/.lab/tmp/s9j_metrics_regen.json >/dev/null 2>&1
if ! diff -q /home/gabe/Documents/gpu-sim/.lab/tmp/s9j_metrics_regen.json metrics.json >/dev/null; then
    echo "metrics.json mismatch"; exit 1
fi
echo "metrics.json recompute matches committed: OK"
echo "DONE"
chmod +x run.sh && ./run.sh
