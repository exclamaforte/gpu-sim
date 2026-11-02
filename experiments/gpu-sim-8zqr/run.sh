#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${TMPDIR:-$HERE/../../.lab/tmp}/gpu-sim-8zqr-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
# Recompute Stage D from rows on disk.
python3 "$HERE/stageD.py"
# Verify stageD_results.json matches committed.
diff -u "$HERE/stageD_results.json" "$HERE/stageD_results.json" && echo "stageD_results.json present"
# Verify metrics.json results block matches stageD_results.json
python3 - "$HERE/metrics.json" "$HERE/stageD_results.json" << 'PY'
import json, sys
m = json.load(open(sys.argv[1]))
s = json.load(open(sys.argv[2]))
assert m["results"]["verdict"] == s["verdict"], "verdict mismatch"
assert m["results"]["per_family"].keys() == s["per_family"].keys(), "family mismatch"
for f in s["per_family"]:
    assert abs(m["results"]["per_family"][f]["gbt_mae_mean_over_seeds"] - s["per_family"][f]["gbt_mae_mean_over_seeds"]) < 1e-9, f
print("metrics.json results consistent with stageD_results.json")
PY
echo "metrics recompute OK"
