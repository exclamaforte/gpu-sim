#!/usr/bin/env bash
# gpu-sim-m3d.2 recompute check. CPU-ONLY.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
SCRATCH="$(mktemp -d /tmp/m3d2-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
make test-fabric > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log" | tail -8
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: make test-fabric exited $rc"; exit 1; }
for t in basic barrier8 killpeer stale size; do
    grep -q "TEST $t: PASS" "$SCRATCH/out.log" || { echo "RECOMPUTE FAIL: $t not PASS"; exit 1; }
done
echo "RECOMPUTE PASS"