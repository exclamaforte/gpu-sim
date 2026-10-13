#!/usr/bin/env bash
# gpu-sim-u2o.6.6 dose (PI 2026-10-06, ruling b) -- JOB BODY, ts ONLY.
# Measures all 124 anchor cells (4 families sequentially) on the idle GPU via
# the 6.6.2 run_family.sh path, so queue_state own_job matching keeps working.
# Outputs mirrored to an absolute out-of-repo dir (worktrees die at merge).
# usage, inside ts: bash dose.sh <outdir>
set -uo pipefail
OUTDIR="${1:?usage: dose.sh <absolute-outdir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAMDIR="$HERE/../gpu-sim-u2o.6.6.2"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUTDIR"
# run_family.sh tees per-family logs into its own dir (relative path); the
# committed anchor logs live there. Back them up first and restore on exit
# so a dose never overwrites committed files (learned the hard way: the
# 20261006 dose clobbered all four and they had to be restored by hand).
BACKUP="$OUTDIR/committed_logs_backup"
mkdir -p "$BACKUP"
for _l in "$FAMDIR"/output_*.log; do cp -f "$_l" "$BACKUP/" 2>/dev/null || true; done
restore_logs() { for _l in "$BACKUP"/output_*.log; do cp -f "$_l" "$FAMDIR/" 2>/dev/null || true; done; }
trap restore_logs EXIT
echo "=== dose start $(date -Is) out=$OUTDIR ==="
export U2O_DOSE_OUTDIR="$OUTDIR"
for FAM in launch memory reduce gemm; do
  echo "--- family=$FAM ---"
  bash "$FAMDIR/run_family.sh" "$FAM" 2>&1 | tee "$OUTDIR/output_${FAM}.log"
  rc=${PIPESTATUS[0]}
  echo "family=$FAM exit=$rc"
  [ "$rc" -eq 0 ] || { echo "DOSE FAIL at $FAM"; exit "$rc"; }
done
sha256sum "$OUTDIR"/family_*.json | tee "$OUTDIR/sha256sums.txt"
echo "=== dose done $(date -Is) ==="
