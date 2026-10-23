#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SCRATCH="${TMPDIR:-/tmp}/gpu-sim-0i8-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

python3 "$HERE/compute_metrics.py" \
  --traces "$REPO/experiments/gpu-sim-pkb/traces" \
  --out "$SCRATCH/metrics.json"

diff -u "$HERE/metrics.json" "$SCRATCH/metrics.json" && echo "metrics.json matches recomputed output"

python3 - "$HERE/metrics.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["decision"] == "CERTIFIED", d["decision"]
assert d["n_qualifying_runs"] == 24, d["n_qualifying_runs"]
assert d["failing_cells"] == [], d["failing_cells"]
assert len(d["runs"]) == 24, len(d["runs"])
print("sanity checks passed")
PY