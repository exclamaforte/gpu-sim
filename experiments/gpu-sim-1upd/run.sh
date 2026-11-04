#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$(dirname "$0")/../.."
mkdir -p .lab/tmp

echo "=== gpu-sim-1upd run.sh ==="

# --- Declared one-time cleanup of pre-existing pycache from pre-fix gate runs ---
before_dirs=$(find experiments -name '__pycache__' -type d | wc -l)
before_pyc=$(find experiments -name '*.pyc' | wc -l)
find experiments -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
find experiments -name '*.pyc' -delete 2>/dev/null || true
after_dirs=$(find experiments -name '__pycache__' -type d | wc -l)
after_pyc=$(find experiments -name '*.pyc' | wc -l)
echo "cleanup: pycache dirs $before_dirs -> $after_dirs; .pyc files $before_pyc -> $after_pyc"

# --- Negative control: without redirect, importing creates .pyc + guard flags it ---
neg_target="experiments/gpu-sim-u2o.17.3"
rm -rf "$neg_target/__pycache__"
env -u PYTHONPYCACHEPREFIX python3 -c \
  "import sys; sys.path.insert(0, '$neg_target'); import sim_trace_173; print('imported')" \
  >.lab/tmp/negctrl.log 2>&1 || true
if [ -d "$neg_target/__pycache__" ]; then
  echo "NEGATIVE CONTROL: without redirect, __pycache__ created under experiments/ (expected)"
  neg_guard=$(git ls-files --others -- experiments/ | grep -c . || true)
  echo "NEGATIVE CONTROL: unfiltered guard query would flag $neg_guard file(s)"
else
  echo "NEGATIVE CONTROL UNEXPECTED: no __pycache__ created without redirect"
  exit 1
fi
rm -rf "$neg_target/__pycache__"
find experiments -name '*.pyc' -delete 2>/dev/null || true

# --- AC1 probe: guard refuses a dir with gitignored __pycache__/*.pyc or *.out ---
probe="experiments/gpu-sim-1upd-probe"
rm -rf "$probe"
mkdir -p "$probe/__pycache__" "$probe/fk"
printf 'x' > "$probe/__pycache__/x.pyc"
printf 'x' > "$probe/fk/probe.out"
echo "PROBE: created $probe with __pycache__/x.pyc and fk/probe.out (both gitignored)"
echo "PROBE wrong-query (unfiltered) flags:"
git ls-files --others -- "$probe"
echo "PROBE correct-query (--exclude-standard) flags: (should be empty)"
git ls-files --others --exclude-standard -- "$probe"
rm -rf "$probe"

# --- Run suite with redirected bytecode cache (conftest + belt-and-braces env) ---
env PYTHONPYCACHEPREFIX=.lab/tmp/pycache python3 -m pytest tests/ -q 2>&1 | tail -5

# --- Repo-wide assertions ---
pc=$(find experiments -name '__pycache__' -type d | wc -l)
pyc=$(find experiments -name '*.pyc' | wc -l)
out=$(find experiments -name '*.out' | wc -l)
echo "post-run: pycache dirs=$pc .pyc files=$pyc .out files=$out"
test "$pc" -eq 0 || { echo "FAIL: pycache dirs under experiments/"; exit 1; }
test "$pyc" -eq 0 || { echo "FAIL: .pyc files under experiments/"; exit 1; }
test "$out" -eq 0 || { echo "FAIL: .out files under experiments/"; exit 1; }

# --- Guard query recompute ---
guard_all=$(git ls-files --others -- experiments/ | wc -l)
echo "guard-query (all untracked, incl. gitignored): $guard_all (informational)"
guard_bad=$(git ls-files --others -- experiments/ | (grep -E '__pycache__|\.pyc$|\.out$' || true) | wc -l)
echo "guard-query (pycache/.pyc/.out only): $guard_bad"
test "$guard_bad" -eq 0 || { echo "FAIL: guard would flag $guard_bad pycache/.pyc/.out file(s)"; git ls-files --others -- experiments/ | grep -E '__pycache__|\.pyc$|\.out$'; exit 1; }

echo "OK: repo-wide clean, guard query empty"
