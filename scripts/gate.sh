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

# gpu-sim-gtree: say WHICH tree this gate is evaluating. With ~80 worktrees under
# .lab/wt/ all sharing this script, a red gate reported by a background session was
# indistinguishable from a stale worktree running an old copy of the checks -- twice in
# one day the lab chased a RED that did not exist in the main tree. Every line of the
# verdict now carries the tree and commit it applies to.
gate_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
gate_head="$(git -C "$gate_root" log -1 --format='%h %s' 2>/dev/null | cut -c1-60)"
echo "GATE TREE: ${gate_root}"
echo "GATE HEAD: ${gate_head:-<no commit>}"
if [ "$gate_root" != "$(cd "$(dirname "$0")/.." && pwd)" ]; then
    echo "GATE WARNING: evaluating a DIFFERENT tree than this script lives in -- a red result"
    echo "              here does not describe the main working copy."
fi

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

RAN=()
run_hard() {  # name, command...
  local name="$1"; shift
  RAN+=("$name")
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
  RAN+=("$name")
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
run_hard "close_audit unchanged" python3 scripts/check_close_audit_unchanged.py
run_hard "close audit (concrete artifacts)" python3 scripts/close_requires_artifact.py
run_hard "interposer guard uniqueness" python3 scripts/check_guard_uniqueness.py
run_hard "libfakegpu freshness" python3 scripts/check_libfakegpu_fresh.py
run_hard "prose-number provenance" python3 scripts/check_prose_numbers.py
run_hard "claims register" python3 scripts/gen_claims_register.py --check
run_hard "no vacuous assertions" python3 scripts/check_no_vacuous_asserts.py
run_advisory "worktree hygiene" python3 scripts/check_worktree_hygiene.py
run_advisory "byte reconciliation" python3 scripts/reconcile_bytes.py --trace experiments/gpu-sim-wrvw/trace_fixture.jsonl --tp 2 --pp 1 --cp 1 --dp 2 --zero 3 --steps 2 --batch 8
run_advisory "byte reconciliation cp" python3 scripts/reconcile_bytes.py --trace experiments/gpu-sim-juzb/trace_fixture.jsonl --tp 1 --pp 1 --cp 2 --dp 1 --zero 0 --steps 2 --batch 8 --cp-grad-mode production

rm -f "$GATE_LOG"

# gpu-sim-gman: THE GATE MUST PROVE IT RAN ITS CHECKS. A check line once went in as
# `run_check` (a function that does not exist); the shell errored, `set -uo pipefail` has no
# -e, and the gate reported GREEN at 6 checks instead of 7. Nothing compared what ran against
# what was supposed to run, so a missing check looks exactly like a passing one. The manifest
# below is that comparison: if a check is dropped, renamed, or silently fails to start, the
# gate now FAILS instead of shrinking.
EXPECTED_HARD=8
EXPECTED_ADVISORY=3
# --fast skips exactly one hard check (the test suite), so the manifest must track the mode
if [ $FAST -eq 1 ]; then
  EXPECTED_TOTAL=$((EXPECTED_HARD - 1 + EXPECTED_ADVISORY))
  EXPECTED_HARD_NOW=$((EXPECTED_HARD - 1))
else
  EXPECTED_TOTAL=$((EXPECTED_HARD + EXPECTED_ADVISORY))
  EXPECTED_HARD_NOW=$EXPECTED_HARD
fi
manifest_fail=0
if [ "${#RAN[@]}" -ne "$EXPECTED_TOTAL" ]; then
  echo "MANIFEST: ${#RAN[@]} checks ran, expected $EXPECTED_TOTAL"
  manifest_fail=1
fi
if [ "$pass" -ne "$EXPECTED_HARD_NOW" ]; then
  echo "MANIFEST: $pass hard checks passed, expected $EXPECTED_HARD_NOW"
  manifest_fail=1
fi
if [ $manifest_fail -ne 0 ]; then
  echo "MANIFEST FAILED: the gate did not run the checks it advertises"
  printf 'ran: %s\n' "${RAN[*]}"
  exit 1
fi

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
