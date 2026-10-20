#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
fail=0
for f in experiments/gpu-sim-m3d.1/negative_controls/*.json; do
  echo "=== $f ==="
  python3 -c "import json,sys; from epic_cluster.spec import validate; validate(json.load(open('$f')))" 2>&1
  rc=$?
  echo "exit=$rc"
  if [ "$rc" -eq 0 ]; then echo "UNEXPECTED SUCCESS"; fail=1; fi
done
exit $fail