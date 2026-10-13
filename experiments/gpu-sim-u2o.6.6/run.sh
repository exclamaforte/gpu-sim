#!/usr/bin/env bash
# gpu-sim-u2o.6.6 -- recompute check. CPU-ONLY (stdlib + repo).
# Re-runs the target gates + target driver into scratch, byte-compares BOTH
# committed artifacts. A diff means the tree moved under the targets (stale
# snapshot); a driver nonzero means a claim stopped holding. Never writes a
# tracked file.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 -m unittest \
    tests.test_noise_targets 2>&1 | tail -n 3
SCRATCH="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/66-recompute-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
CUDA_VISIBLE_DEVICES= nice -n 10 python3 "$HERE/build_targets.py" \
    --targets-out "$SCRATCH/targets.jsonl" \
    --metrics-out "$SCRATCH/metrics.json" > "$SCRATCH/out.log" 2>&1
rc=$?
tail -n 1 "$SCRATCH/out.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: driver rc=$rc"; exit 1; }
cmp -s "$HERE/targets.jsonl" "$SCRATCH/targets.jsonl" \
    || { echo "RECOMPUTE FAIL: targets differ"; exit 1; }
cmp -s "$HERE/metrics.json" "$SCRATCH/metrics.json" \
    && echo "RECOMPUTE PASS (byte-identical)" \
    || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
