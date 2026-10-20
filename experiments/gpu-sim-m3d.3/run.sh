#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
SCRATCH="$(mktemp -d /tmp/m3d3-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
python3 -m pytest tests/test_link_model.py -q > "$SCRATCH/out.log" 2>&1
rc=$?
cat "$SCRATCH/out.log" | tail -3
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: pytest exited $rc"; exit 1; }
echo "RECOMPUTE PASS"
