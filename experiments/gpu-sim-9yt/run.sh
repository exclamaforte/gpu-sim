#!/usr/bin/env bash
# gpu-sim-9yt RUN (CPU, unqueued): regen proof + single-definition suite.
# proof.json/metrics.json replay byte-identical (deterministic JSON);
# any mismatch fails instead of landing.
set -euo pipefail
HERE="$(dirname "$0")"
cd "$HERE"
export CUDA_VISIBLE_DEVICES=

snap() { sha256sum proof.json metrics.json | sort; }
before=$(mktemp); snap > "$before"
python3 regen_proof.py > /tmp/9yt_run_regen.log 2>&1
tail -3 /tmp/9yt_run_regen.log
cd ../..
python3 tests/test_kernel_hash_single_definition.py 2>&1 | tail -4
cd experiments/gpu-sim-9yt
after=$(mktemp); snap > "$after"
if ! diff -q "$before" "$after" > /dev/null; then
  echo "RUN FAIL: replay changed committed artifacts:"; diff "$before" "$after"
  exit 1
fi
echo "RUN PASS: 9yt regen + single-definition suite green, replay byte-identical"
