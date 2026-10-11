#!/usr/bin/env bash
# gpu-sim-u2o.9.4 -- recompute check. CPU-ONLY (stdlib + repo).
# Re-runs the report gates + report driver, byte-compares metrics.json.
# NOT a model gate: nothing under epic_u2o/ may change in this bead, so a
# metrics diff means the tree moved under the report (stale snapshot), and
# a driver nonzero means a claim stopped holding.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 -m unittest \
    tests.test_baseline_report 2>&1 | tail -n 3
SCRATCH="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/94-recompute-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
cp "$HERE/metrics.json" "$SCRATCH/metrics.committed"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/report_verify.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
tail -n 2 "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: driver rc=$rc"; exit 1; }
cmp -s "$SCRATCH/metrics.committed" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
