#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE:${PYTHONPATH:-}"
# verify all m1 traces exist
for S in 4096 65536 131072 262144 393216 524288 1048576 2097152; do
  test -s "traces/trace_S${S}_n2_m1.log" || { echo "missing m1 trace for $S"; exit 1; }
done
# rebuild table
python3 build_table.py > /tmp/scratchpad/eut_table.log 2>&1
test -s table_results.json
# negative controls must be rejected (exit 2)
if python3 reduce_trace.py traces/negative_control_broken.log >/dev/null 2>&1; then
  echo "FAIL: broken trace was accepted"; exit 1
fi
if python3 reduce_trace.py traces/negative_control_nonmultiple.log >/dev/null 2>&1; then
  echo "FAIL: nonmultiple trace was accepted"; exit 1
fi
echo "run.sh OK: table rebuilt, negative controls rejected"
