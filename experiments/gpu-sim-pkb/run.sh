#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
# Verify all canonical traces exist
for f in traces/trace_S4096_n2_eut4-4096-1792598352.log traces/trace_S65536_n2_CLEAN.log traces/trace_S262144_n2_CLEAN.log traces/trace_S524288_n2_CLEAN.log traces/trace_S1048576_n2_CLEAN.log traces/trace_S2097152_n2_CLEAN.log; do
  test -s "$f" || { echo "missing trace: $f"; exit 1; }
done
# Rebuild table
python3 build_table.py > /tmp/scratchpad/eut_table.log 2>&1
test -s table_results.json
echo "Table rebuilt; see table_results.json"
# Verify completeness: no valid trace shows factor 2.0
python3 -c "
import json
d=json.load(open('table_results.json'))
for r in d['rows']:
    if r['valid']:
        for rk in r['ranks']:
            assert rk['byte_factor'] < 2.0, f\"unexpected 2.0 factor at {r['S_bytes']}\"
print('NO-STEP verified: no valid trace has byte factor 2.0')
"
chmod +x /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-eut/experiments/gpu-sim-pkb/run.sh
cd /home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-eut/experiments/gpu-sim-pkb && ./run.sh
