#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${TMPDIR:-$HERE/../../.lab/tmp}/gpu-sim-8zqr-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
# Recompute Stage D from rows on disk into a scratch path (never overwrite committed).
python3 "$HERE/stageD.py" --out "$SCRATCH/stageD_results.fresh.json"
# Self-diff against committed.
if ! diff -u "$HERE/stageD_results.json" "$SCRATCH/stageD_results.fresh.json"; then
  echo "stageD_results.json NOT reproducible"; exit 1; fi
echo "stageD_results.json reproducible"
# Verify metrics.json results consistent.
python3 - "$HERE/metrics.json" "$SCRATCH/stageD_results.fresh.json" << 'PY'
import json, sys
m = json.load(open(sys.argv[1]))
s = json.load(open(sys.argv[2]))
assert m["results"]["verdict"] == s["verdict"], "verdict mismatch"
for f in s["per_family"]:
    assert abs(m["results"]["per_family"][f]["gbt_mae_mean_over_seeds"] - s["per_family"][f]["gbt_mae_mean_over_seeds"]) < 1e-9, f
print("metrics.json results consistent with stageD_results.json")
PY
echo "metrics recompute OK"
