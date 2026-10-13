#!/usr/bin/env bash
# gpu-sim-0vk RUN (CPU, unqueued): replay merge + census + OOB-naming + verify.
# Fails on any mismatch: outputs must be byte-identical to the committed
# artifacts (checked by sha256 before/after), and verify must exit 0.
# Never writes a tracked file with new content: a replay that changes any
# artifact fails instead of landing.
set -euo pipefail
HERE="$(dirname "$0")"
cd "$HERE"

snap() { sha256sum measured_new.jsonl merge_summary.json shortfall.json census.json oob_frozen.json metrics.json scratch_neg/neg_control.json | sort; }
export CUDA_VISIBLE_DEVICES=

before=$(mktemp); snap > "$before"
python3 merge_0vk.py > /tmp/0vk_run_merge.log 2>&1
python3 census_0vk.py > /tmp/0vk_run_census.log 2>&1
python3 name_oob_rows.py > /tmp/0vk_run_oob.log 2>&1
python3 verify_0vk.py > /tmp/0vk_run_verify.log 2>&1
after=$(mktemp); snap > "$after"

tail -3 /tmp/0vk_run_merge.log
tail -4 /tmp/0vk_run_census.log
tail -2 /tmp/0vk_run_oob.log
tail -2 /tmp/0vk_run_verify.log
if ! diff -q "$before" "$after" > /dev/null; then
  echo "RUN FAIL: replay changed committed artifacts:"; diff "$before" "$after"
  exit 1
fi
echo "RUN PASS: merge + census + oob-naming + verify replay byte-identical"
