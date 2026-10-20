#!/usr/bin/env bash
# Freshness assertion for the primary evidence logs. A check only ever seen to
# pass is not a check, so this is demonstrated RED on the stale logs kept under
# stale/ before it is claimed GREEN on the fresh ones.
set -uo pipefail
rc=0
for f in "$@"; do
  [ -f "$f" ] || { echo "MISSING: $f"; rc=1; continue; }
  if grep -q "dlsym fallback" "$f"; then
    echo "STALE: $f contains 'dlsym fallback' (removed from src/fakecuda.c at 7cc9d54)"; rc=1
  fi
  if ! grep -q "AUDIT" "$f"; then
    echo "STALE: $f has no AUDIT lines (pre-audit revision)"; rc=1
  fi
done
[ "$rc" -eq 0 ] && echo "FRESH: all logs contain AUDIT lines and none contains 'dlsym fallback'"
exit "$rc"
