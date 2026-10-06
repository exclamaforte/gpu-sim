#!/usr/bin/env bash
# gpu-sim-u2o.6.6 -- recompute check. CPU-ONLY: statistics over committed JSON,
# no CUDA, no ts queue. Re-runs build_targets.py into a scratch dir and
# byte-compares against the committed metrics.json (seeded RNG + sort_keys make
# it deterministic). Writes nothing tracked. Fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$HERE"
SCRATCH="$(mktemp -d /tmp/u66-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/build_targets.py" \
    --metrics-out "$SCRATCH/recomputed.json" > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: driver exited $rc"; exit 1; }
cmp -s "$HERE/metrics.json" "$SCRATCH/recomputed.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; diff <(head -c 2000 "$HERE/metrics.json") <(head -c 2000 "$SCRATCH/recomputed.json"); exit 1; }
