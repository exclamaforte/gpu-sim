#!/usr/bin/env bash
# gpu-sim-u2o.8.2 RUN (CPU, unqueued): replay verification from the artifact.
# The 40-min extraction is a recorded run (see job.json), NOT replayed here;
# verify recomputes every check from ptx_corpus.jsonl + the corpus.
set -euo pipefail
HERE="$(dirname "$0")"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=
snap() { sha256sum metrics.json | sort; }
before=$(mktemp); snap > "$before"
python3 verify_82.py > /tmp/82_run_verify.log 2>&1
tail -3 /tmp/82_run_verify.log
after=$(mktemp); snap > "$after"
if ! diff -q "$before" "$after" > /dev/null; then
  echo "RUN FAIL:"; diff "$before" "$after"; exit 1
fi
echo "RUN PASS: 8.2 verify green, metrics stable"
