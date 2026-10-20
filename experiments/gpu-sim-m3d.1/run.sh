#!/usr/bin/env bash
# gpu-sim-m3d.1 recompute check. CPU-ONLY.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
SCRATCH="$(mktemp -d /tmp/m3d1-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT

python3 -m pytest tests/test_cluster_spec.py -q > "$SCRATCH/pytest.log" 2>&1
rc=$?
cat "$SCRATCH/pytest.log" | tail -3
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: pytest exited $rc"; exit 1; }

bash experiments/gpu-sim-m3d.1/run_negative_controls.sh > "$SCRATCH/nc.log" 2>&1
rc=$?
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: negative controls failed"; cat "$SCRATCH/nc.log"; exit 1; }

echo "RECOMPUTE PASS"