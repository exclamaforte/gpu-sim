#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH="${TMPDIR:-$HERE/../../.lab/tmp}/gpu-sim-8zqr-metrics.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
# Recompute Stage D from rows on disk into a scratch path (never overwrite committed).
python3 "$HERE/stageD.py" --out "$SCRATCH/stageD_results.fresh.json"
# Self-diff against committed.
if ! diff -u "$HERE/stageD_results.json" "$SCRATCH/stageD_results.fresh.json"; then
  echo "stageD_results.json NOT reproducible"; exit 1; fi
echo "stageD_results.json reproducible"
# Recursive deep-compare: every path in stageD_results.json must exist in
# metrics.json['results'] with an equal value (floats within 1e-9).
python3 - "$HERE/metrics.json" "$SCRATCH/stageD_results.fresh.json" << 'PY'
import json, sys, math

def norm(v):
    if isinstance(v, float):
        return v
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(vv) for k, vv in v.items()}
    return v

def eq(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(eq(a[k], b[k]) for k in a)
    return a == b

m = json.load(open(sys.argv[1]))["results"]
s = json.load(open(sys.argv[2]))

def walk(spath, sval, mroot, mpath):
    if isinstance(sval, dict):
        if not isinstance(mroot, dict) or spath.split(".")[-1] not in mroot and spath != "":
            pass
        for k, v in sval.items():
            mp = f"{mpath}.{k}" if mpath else k
            if not isinstance(mroot, dict) or k not in mroot:
                print(f"MISMATCH: path '{mp}' present in stageD_results but missing in metrics.json results")
                return False
            if not walk(mp, v, mroot[k], mp):
                return False
        return True
    if isinstance(sval, list):
        if not isinstance(mroot, list) or len(sval) != len(mroot):
            print(f"MISMATCH: path '{mpath}' list length differs ({len(sval)} vs {len(mroot) if isinstance(mroot,list) else 'n/a'})")
            return False
        for i, (sv, mv) in enumerate(zip(sval, mroot)):
            if not walk(f"{mpath}[{i}]", sv, mv, f"{mpath}[{i}]"):
                return False
        return True
    # scalar
    if not eq(sval, mroot):
        print(f"MISMATCH: path '{mpath}' differs: stageD={sval!r} vs metrics={mroot!r}")
        return False
    return True

if not walk("", s, m, ""):
    sys.exit(1)
# also check no extra top-level keys? Not required; only stageD paths must be in metrics.
print("metrics.json results deep-consistent with stageD_results.json")
PY
echo "metrics recompute OK"
