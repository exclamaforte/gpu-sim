#!/usr/bin/env bash
# gpu-sim-ep3 orchestrator. Invoke under the 8-CPU quota so the windows inherit it:
#   systemd-run --user --scope --property=CPUQuota=800% -- bash experiments/gpu-sim-ep3/run.sh
set -u
cd "$(dirname "$0")"

echo "== selection (5s sample of all CPUs) =="
python3 select_cpu.py > selection.json
cat selection.json

TARGET=$(python3 -c "import json;print(json.load(open('selection.json'))['target'])")
UTIL=$(python3 -c "import json;print(json.load(open('selection.json'))['target_util'])")

if python3 -c "import sys; sys.exit(0 if float('$UTIL') <= 0.25 else 1)"; then
  echo "== target cpu=$TARGET util=$UTIL (<=0.25) -- running 3 windows =="
  for w in 1 2 3; do
    echo "-- window $w --"
    taskset -c "$TARGET" python3 measure_window.py --cpu "$TARGET" --window "$w" > "window_${w}.out" 2>&1
    tail -3 "window_${w}.out"
  done
else
  echo "== NO-QUIET-CORE: least-busy cpu=$TARGET has util=$UTIL > 0.25 -- NO WINDOWS RUN =="
  printf '{"verdict":"NO-QUIET-CORE","target_cpu":%s,"target_util":%s}\n' "$TARGET" "$UTIL" > verdict.json
fi
echo "== done =="
