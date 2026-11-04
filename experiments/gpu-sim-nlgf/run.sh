#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export U5R4_OUT="${TMPDIR:-/tmp}/nlgf_out"
rm -rf "$U5R4_OUT"; mkdir -p "$U5R4_OUT"
python3 "$HERE/analyze_power.py"
if ! diff -q "$U5R4_OUT/metrics.json" "$HERE/metrics.json" >/dev/null; then
    echo "METRICS_MISMATCH"; diff "$U5R4_OUT/metrics.json" "$HERE/metrics.json"; exit 1
fi
echo "METRICS_MATCH"
