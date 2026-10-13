#!/usr/bin/env bash
# gpu-sim-u2o.6.5.1 -- recompute check. CPU-ONLY: no CUDA, no ts queue.
# Re-runs the new isolation checks + make check, and re-verifies the bead's
# standing admission grep (every GPU-free claim has a named check).
# Writes nothing tracked; fails on mismatch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
SCRATCH="$(mktemp -d /tmp/u651-recompute-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT
export CUDA_VISIBLE_DEVICES=
echo "--- tests/test_isolation_claims.py ---"
nice -n 10 python3 tests/test_isolation_claims.py > "$SCRATCH/claims.log" 2>&1
rc=$?
cat "$SCRATCH/claims.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: isolation claims exited $rc"; exit 1; }
N_PASS=$(grep -c "^PASS" "$SCRATCH/claims.log")
[ "$N_PASS" -eq 17 ] || { echo "RECOMPUTE FAIL: expected 17 PASS, got $N_PASS"; exit 1; }
echo "--- make check ---"
nice -n 10 make check > "$SCRATCH/makecheck.log" 2>&1
rc=$?
tail -n 12 "$SCRATCH/makecheck.log"
[ "$rc" -eq 0 ] || { echo "RECOMPUTE FAIL: make check exited $rc"; exit 1; }
echo "--- standing grep: GPU-free claims ---"
grep -rni "GPU-free" epic_u2o/*.py shim/*.h | tee "$SCRATCH/grep.log"
# Every remaining GPU-free claim must be one the inventory backs (rows 4-6).
grep -q "preflight_validator.py" "$SCRATCH/grep.log" || { echo "RECOMPUTE FAIL: preflight claim moved"; exit 1; }
grep -q "safe_harness.py" "$SCRATCH/grep.log" || { echo "RECOMPUTE FAIL: harness claim moved"; exit 1; }
echo "RECOMPUTE PASS"
