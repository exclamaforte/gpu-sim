#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SCRATCH="${TMPDIR:-/tmp}/gpu-sim-bk9-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

python3 "$HERE/compute_metrics.py" --traces "$HERE/traces" --out "$SCRATCH/metrics.json"
diff -u "$HERE/metrics.json" "$SCRATCH/metrics.json" && echo "metrics.json matches recomputed output"

python3 - "$HERE/metrics.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert len(d["cells"]) == 25, len(d["cells"])
assert "usable_count" in d
print("sanity checks passed")
PY