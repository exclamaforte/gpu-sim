#!/usr/bin/env bash
# gpu-sim-u2o.14 RUN (CPU, unqueued): replay dose merge byte-identical.
set -euo pipefail
HERE="$(dirname "$0")"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=
snap() { sha256sum dose_measured.jsonl dose_summary.json metrics.json | sort; }
# metrics.json is merge-generated: absent on a fresh checkout, so merge once
# before snapshotting, then assert the second run is byte-identical.
python3 merge_dose_14.py > /tmp/14_run_merge.log 2>&1
tail -2 /tmp/14_run_merge.log
before=$(mktemp); snap > "$before"
python3 merge_dose_14.py >> /tmp/14_run_merge.log 2>&1
python3 -c "
import json
s = json.load(open('dose_summary.json'))
m = json.load(open('metrics.json'))
assert m['median_ai_cv'] == s['median_ai_cv'] and m['n_measured'] == s['n_measured'], (m, s)
print('metrics anchored to dose_summary')
"
after=$(mktemp); snap > "$after"
if ! diff -q "$before" "$after" > /dev/null; then
  echo "RUN FAIL:"; diff "$before" "$after"; exit 1
fi
echo "RUN PASS: 14 dose merge replay byte-identical"
