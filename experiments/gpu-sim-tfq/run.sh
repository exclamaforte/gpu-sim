#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1. statistic agreement with verdict.py on the same raw log
python3 check_statistic_agreement.py >/tmp/tfq_agree.log 2>&1
echo "statistic agreement: OK"

# 2. recompute every window's spread from its raw log; must match windows.jsonl
python3 - <<'PY'
import json, os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spread import spread_us
rows=[json.loads(l) for l in open(os.path.join(HERE,"windows.jsonl"))]
for r in rows:
    raw=os.path.join(HERE, "raw", os.path.basename(r["raw_log"]))
    vals=[float(x) for x in open(raw) if x.strip()]
    recomputed=spread_us(vals)
    if abs(recomputed-r["spread_us"])>0.15:
        print("MISMATCH", r["window_id"], recomputed, r["spread_us"]); raise SystemExit(1)
print("all", len(rows), "windows recompute OK")
PY

# 3. regenerate metrics.json into scratch and diff against committed
python3 build_metrics.py /tmp/tfq_metrics_regen.json >/tmp/tfq_build.log 2>&1
if ! diff -q /tmp/tfq_metrics_regen.json metrics.json >/dev/null; then
    echo "metrics.json mismatch after recompute"; diff /tmp/tfq_metrics_regen.json metrics.json | head; exit 1
fi
echo "metrics.json recompute matches committed: OK"

echo "DONE"
