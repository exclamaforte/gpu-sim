#!/usr/bin/env bash
# gpu-sim-mgxy: reproduce the NULL-cell measurement (why a kernel can measure
# faster than the empty kernel) with the warm-min floor as the fix.
#
# WHAT IS MEASURED, and nothing else: the SAME empty binary is run twice and the
# two series are treated as (scale, empty).  A null cell cannot be faster than
# itself, so any non-zero reading there is the estimator's own noise.  The arm
# reports, over NREP repeats, how often each estimator returns a negative
# kernel_us for the null cell and for a real tiny kernel.
#
# SUBSTITUTED INPUT, declared rather than hidden: the provenance log was
# produced with prebuilt 8zqr binaries including empty_16_1_1_64_1_1.out (grid
# 16x64).  That build directory is not in the tree; build.py regenerates the
# baseline for the divergent_branch family as grid 4x64.  This arm therefore
# measures the estimator on the pair THIS tree builds, and its numbers are not
# element-wise comparable to the old log.
#
# GPU: the binaries launch CUDA kernels, so this is a GPU-queue job.
set -euo pipefail

# One thread per numeric library (tests/test_pin_coverage.py).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=${GPU_SIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
EXP="$ROOT/experiments/gpu-sim-mgxy"
OUT=${OUTMGXY:-"$ROOT/.lab/tmp/mgxy-$(date +%Y%m%dT%H%M%S)"}
NREP=${NREP:-20}
mkdir -p "$OUT"

fail() { echo "MGXY FAIL: $*"; exit 1; }

echo "--- building the two binaries this arm uses (CPU phase)"
python3 "$ROOT/experiments/gpu-sim-8zqr/build.py" \
    --families divergent_branch --n-per-family 1 >"$OUT/build.log" 2>&1 \
    || { tail -5 "$OUT/build.log"; fail "build.py failed"; }
BIN="$ROOT/experiments/gpu-sim-8zqr/build/bin"
EMPTY=$(ls "$BIN"/empty_4_1_1_64_1_1.out 2>/dev/null || ls "$BIN"/empty_*.out | head -1)
SCALE="$BIN/k_divergent_branch_0000.out"
[[ -x "$EMPTY" ]] || fail "no empty baseline built"
[[ -x "$SCALE" ]] || fail "no scale kernel built"
echo "MGXY empty_binary=$(basename "$EMPTY")"
echo "MGXY scale_binary=$(basename "$SCALE")"

echo "--- measuring ($NREP repeats)"
t0=$(date +%s.%N)
GPU_SIM_8ZQR_BIN="$BIN" GPU_SIM_EMPTY_BIN="$(basename "$EMPTY")" \
    python3 "$EXP/instrument_null.py" --repeats "$NREP" >"$OUT/null_cell.log" 2>&1 \
    || true
t1=$(date +%s.%N)
wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')

python3 - "$OUT/null_cell.log" "$OUT/summary.json" "$NREP" "$wall" <<'PY'
import json, re, sys

log, out_json, nrep, wall = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
rows = []
for line in open(log):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    if "null_median_us" in d:
        rows.append(d)

def neg(vals):
    return sum(1 for x in vals if x < 0)

def med(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

out = {"n_repeats": len(rows), "requested_repeats": nrep, "wall_s": wall}
if len(rows) != nrep:
    print(f"MGXY WARNING: parsed {len(rows)} of {nrep} repeats")
for name, key in (("null_median", "null_median_us"),
                  ("null_warm_min", "null_warm_min_us"),
                  ("real_median", "real_median_us"),
                  ("real_warm_min", "real_warm_min_us")):
    vals = [r[key] for r in rows]
    out[name] = {"median_us": round(med(vals), 3), "min_us": round(min(vals), 3),
                 "max_us": round(max(vals), 3), "negative": neg(vals),
                 "n": len(vals)}
json.dump(out, open(out_json, "w"), indent=2)
print(json.dumps(out, indent=2))
PY

fails=0
chk() { if eval "$1"; then echo "ASSERT ok: $2"; else echo "ASSERT FAIL: $2"; fails=$((fails+1)); fi; }
chk "python3 -c \"import json,sys; d=json.load(open('$OUT/summary.json')); sys.exit(0 if d['null_median']['negative'] >= 1 else 1)\"" \
    "the NULL cell reads NEGATIVE under the shipped median estimator (a kernel cannot be faster than itself); the RATE is baseline-dependent and reported, not asserted"
chk "python3 -c \"import json,sys; d=json.load(open('$OUT/summary.json')); sys.exit(0 if d['null_median']['min_us'] < 0 else 1)\"" \
    "the null cell's worst reading is NEGATIVE in us"
chk "python3 -c \"import json,sys; d=json.load(open('$OUT/summary.json')); sys.exit(0 if d['null_warm_min']['negative'] <= 1 else 1)\"" \
    "the warm-min floor removes the null-cell negatives (<=1/$NREP)"
chk "python3 -c \"import json,sys; d=json.load(open('$OUT/summary.json')); sys.exit(0 if d['real_warm_min']['negative'] == 0 else 1)\"" \
    "the same fix leaves the REAL cell non-negative (0/$NREP) -- it does not buy the result with a false negative"
chk "[[ -s '$OUT/null_cell.log' ]]" "the raw per-repeat log exists"

echo "MGXY wall_s=$wall"
echo "MGXY tip=$(git -C "$ROOT" rev-parse HEAD)"
echo "OUT=$OUT"
echo
if (( fails )); then
    echo "MGXY RUN FAILED: $fails assertion(s); artifacts kept at $OUT"
    exit 1
fi
echo "MGXY RUN OK: null cell negative under the median estimator, floored by warm-min, real cell unaffected"
echo "OUT=$OUT"
