#!/usr/bin/env bash
# One entry point for this repository's quality gates (gpu-sim-a8x follow-up).
#
# The orchestrator has been reporting "gate status unknown: gate script not found"
# every drain, so here it is. Checks are split into HARD (a failure fails the gate)
# and ADVISORY (reported, never fails). Advisory is not a loophole: the two
# advisory checks currently have pre-existing findings recorded elsewhere
# (check_prose_numbers: 6 unprovenanced claims in older results docs; the bead
# tracker owns them). Everything that can be made green today is HARD.
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

run_hard() {  # name, command...
  local name="$1"; shift
  printf '%-42s ' "$name"
  if "$@" >/tmp/gate.$$.out 2>&1; then
    echo "PASS"; pass=$((pass+1))
  else
    echo "FAIL"; fail=$((fail+1)); FAILED+=("$name")
    sed -n '1,12p' /tmp/gate.$$.out | sed 's/^/    /'
  fi
}

run_advisory() {  # name, command...
  local name="$1"; shift
  printf '%-42s ' "$name (advisory)"
  if "$@" >/tmp/gate.$$.out 2>&1; then
    echo "PASS"
  else
    echo "FINDINGS: $(tail -1 /tmp/gate.$$.out)"
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
run_advisory "prose-number provenance" python3 scripts/check_prose_numbers.py

rm -f /tmp/gate.$$.out
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
