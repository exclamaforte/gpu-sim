#!/usr/bin/env bash
# gpu-sim-edu.2 -- CPU only, run directly. Never through ts.
# Inventories the lab runtime (read-only, git asked via dry run) and proves the
# guard refuses anything that would delete it.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-edu.2"
echo "=== lab runtime inventory: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 "$ART/inventory.py" 2>&1 | tee "$ART/output.log" | tail -14
rc=${PIPESTATUS[0]}; echo "=== inventory exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"

MAIN=$(git rev-parse --path-format=absolute --git-common-dir); MAIN=$(dirname "$MAIN")
echo "=== guard verdict on 'git clean -fd' against the MAIN repo ==="
python3 scripts/guard_destructive_git.py --repo "$MAIN" --check-only -- clean -fd \
  2>&1 | tee -a "$ART/output.log" | head -2
python3 scripts/guard_destructive_git.py --repo "$MAIN" --check-only -- clean -fd \
  >/dev/null 2>&1; grc=$?
echo "=== guard exit=$grc (2 = REFUSED, which is the point) ==="
[ "$grc" -eq 2 ] || { echo "GUARD DID NOT REFUSE"; exit 1; }

echo "=== tests ==="
python3 tests/test_destructive_git_guard.py 2>&1 | tee -a "$ART/output.log" | tail -4
rc2=${PIPESTATUS[0]}; echo "=== tests exit=$rc2 ==="; [ "$rc2" -eq 0 ] || exit "$rc2"
echo "=== run.sh: all green ==="
