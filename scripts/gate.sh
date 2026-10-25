#!/usr/bin/env bash
# One entry point for this repository's quality gates (gpu-sim-a8x follow-up).
#
# The orchestrator has been reporting "gate status unknown: gate script not found"
# every drain, so here it is. Checks are split into HARD (a failure fails the gate)
# and ADVISORY (reported, never fails). Advisory is not a loophole: the two
# check_prose_numbers promoted from ADVISORY to HARD (gpu-sim-xte): all
# findings either cited or grandfathered.
#
# Usage:
#   scripts/gate.sh            # full: pytest (~2 min) + every check
#   scripts/gate.sh --fast     # skip the test suite (docs/model/audit checks only)
set -uo pipefail
cd "$(dirname "$0")/.."

FAST=0
[ "${1:-}" = "--fast" ] && FAST=1

pass=0; fail=0; advisory=0
declare -a FAILED=() ADVISED=()

# gpu-sim-smt5: the gate used to buffer each check's output in /tmp. On a host where /tmp is
# out of quota (this one: 13G tmpfs with a project quota, routinely exhausted by trace dumps)
# EVERY check fails with an empty log and no diagnostics -- the gate went red with 0/6 passing
# while each check passed when run directly. Keep the log inside the repo, and if the redirect
# itself fails, say so instead of reporting a phantom check failure.
GATE_TMP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.lab/tmp"
mkdir -p "$GATE_TMP" || { echo "gate: cannot create $GATE_TMP" >&2; exit 1; }
GATE_LOG="$GATE_TMP/gate.$$.out"
if ! : > "$GATE_LOG" 2>/dev/null; then
  echo "gate: cannot write $GATE_LOG (filesystem full or read-only)" >&2
  exit 1
fi

run_hard() {  # name, command...
  local name="$1"; shift
  printf '%-42s ' "$name"
  if "$@" >"$GATE_LOG" 2>&1; then
    echo "PASS"; pass=$((pass+1))
  else
    echo "FAIL"; fail=$((fail+1)); FAILED+=("$name")
    if [ -s "$GATE_LOG" ]; then
      sed -n '1,12p' "$GATE_LOG" | sed 's/^/    /'
    else
      echo "    (no output: check produced no log -- did it fail to start?)"
    fi
  fi
}

run_advisory() {  # name, command...
  local name="$1"; shift
  printf '%-42s ' "$name (advisory)"
  if "$@" >"$GATE_LOG" 2>&1; then
    echo "PASS"
  else
    echo "FINDINGS: $(tail -1 "$GATE_LOG")"
    advisory=$((advisory+1)); ADVISED+=("$name")
  fi
}

echo "=== gpu-sim gates $(date -u +%Y-%m-%dT%H:%MZ) ==="
[ $FAST -eq 1 ] && echo "(fast mode: test suite skipped)"

if [ $FAST -eq 0 ]; then
  run_hard "pytest tests/" python3 -m pytest tests/ -q
fi
run_hard "docs section 0 vs model" python3 scripts/check_docs_model_numbers.py
run_hard "close audit (concrete artifacts)" python3 scripts/close_requires_artifact.py
run_hard "interposer guard uniqueness" python3 scripts/check_guard_uniqueness.py
run_hard "libfakegpu freshness" python3 scripts/check_libfakegpu_fresh.py
run_hard "prose-number provenance" python3 scripts/check_prose_numbers.py

rm -f "$GATE_LOG"
echo
echo "hard: $pass passed, $fail failed | advisory findings: $advisory"
if [ $fail -ne 0 ]; then
  printf 'FAILED: %s\n' "${FAILED[*]}"
  exit 1
fi
if [ $advisory -ne 0 ]; then
  printf 'advisory (pre-existing, tracked elsewhere): %s\n' "${ADVISED[*]}"
fi
echo "GATE OK"
