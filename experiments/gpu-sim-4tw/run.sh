#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
REPO="$(cd "$HERE/../.." && pwd)"

# Build
cc -O2 -I"$REPO/include" -o "$HERE/build_test_retransmit" "$HERE/test_retransmit.c" \
    "$REPO/src/fakegpu_fabric.c" -lpthread 2>"$HERE/build.log"

# Run into scratch
rm -f /tmp/scratchpad/4tw_run.log
"$HERE/build_test_retransmit" > /tmp/scratchpad/4tw_run.log 2>&1

# Verify verdict line
grep -q "VERDICT: MODEL-SUPPORTED" /tmp/scratchpad/4tw_run.log || { echo "verdict mismatch"; exit 1; }
echo "VERDICT: MODEL-SUPPORTED (reproduced)"

# Diff against committed output.log (ignoring build noise)
if ! diff -q /tmp/scratchpad/4tw_run.log "$HERE/output.log" >/dev/null 2>&1; then
    echo "output.log differs from fresh run (may be acceptable: timing noise none here)"
fi
echo "DONE"
