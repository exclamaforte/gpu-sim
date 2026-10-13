#!/usr/bin/env bash
# gpu-sim-u2o.14 RUN (CPU, unqueued): replay dose merge byte-identical.
set -euo pipefail
HERE="$(dirname "$0")"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=
snap() { sha256sum dose_measured.jsonl dose_summary.json | sort; }
before=$(mktemp); snap > "$before"
python3 merge_dose_14.py > /tmp/14_run_merge.log 2>&1
tail -2 /tmp/14_run_merge.log
after=$(mktemp); snap > "$after"
if ! diff -q "$before" "$after" > /dev/null; then
  echo "RUN FAIL:"; diff "$before" "$after"; exit 1
fi
echo "RUN PASS: 14 dose merge replay byte-identical"
