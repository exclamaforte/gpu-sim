#!/usr/bin/env bash
# gpu-sim-xy8 -- the negative-control demonstration, standing practice.
# CPU-ONLY. Never writes a tracked file (scratch under $XDG_CACHE_HOME,
# builds into gitignored build//src/*.o).
#
# Gates (deterministic):
#  1. clean full rebuild of libfakegpu.so from zero .o files: 0 warnings.
#  2. a scratch-broken fg_handle_find_name (miss return removed, i.e. the
#     exact UB that shipped in 0837970) FAILS to build WITH -Werror=return-type
#     and BUILDS with the flag stripped (warning only).
# Evidence (recorded, not gated -- UB codegen luck is not deterministic):
#  3. the miss test run against the no-flag broken build passes exit 0
#     ("ok miss returns NULL, not garbage"): green by rax luck, which is
#     the finding this bead exists to record.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=
SCRATCH="$(mktemp -d "${XDG_CACHE_HOME:-$HOME/.cache}/xy8-recompute-XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT
mkdir -p build
exec_out="$SCRATCH/out.log"
: > "$exec_out"

echo "--- 1. clean rebuild, warning count ---" | tee -a "$exec_out"
rm -f src/*.o
nice -n 10 make > "$SCRATCH/make.log" 2>&1
make_rc=$?
warns=$(grep -ci "warning" "$SCRATCH/make.log" || true)
echo "make rc=$make_rc warnings=$warns" | tee -a "$exec_out"
[ "$make_rc" -eq 0 ] || { echo "FAIL: rebuild broken"; exit 1; }
[ "$warns" -eq 0 ] || { echo "FAIL: $warns warnings, policy not free"; exit 1; }

echo "--- 2. negative control: broken find_name, with and without the flag ---" | tee -a "$exec_out"
mkdir -p "$SCRATCH/broken" && cp src/fakegpu_pool.c "$SCRATCH/broken/"
python3 - "$SCRATCH/broken/fakegpu_pool.c" <<'EOF'
import sys
p = sys.argv[1]
lines = open(p).read().splitlines()
drop = [i for i, l in enumerate(lines)
        if l.strip().startswith("return NULL; /* miss:")]
assert len(drop) == 1, f"miss-return block not found once: {drop}"
del lines[drop[0]:drop[0] + 3]  # return + 2 comment continuation lines
open(p, "w").write("\n".join(lines) + "\n")
print(f"removed miss return at line {drop[0] + 1}")
EOF
grep -c "return NULL" "$SCRATCH/broken/fakegpu_pool.c" | tee -a "$exec_out"
CFLAGS_BASE="-O2 -g -fPIC -Wall -Wextra -Wno-unused-parameter"
nice -n 10 gcc $CFLAGS_BASE -Werror=return-type -Iinclude -Ishim \
    -c "$SCRATCH/broken/fakegpu_pool.c" -o "$SCRATCH/broken_ng.o" \
    > "$SCRATCH/flag.log" 2>&1
with_flag_rc=$?
echo "with flag rc=$with_flag_rc (want nonzero)" | tee -a "$exec_out"
grep -m1 "control reaches end" "$SCRATCH/flag.log" | tee -a "$exec_out"
nice -n 10 gcc $CFLAGS_BASE -Iinclude -Ishim \
    -c "$SCRATCH/broken/fakegpu_pool.c" -o "$SCRATCH/broken_ok.o" \
    > "$SCRATCH/noflag.log" 2>&1
without_flag_rc=$?
echo "without flag rc=$without_flag_rc (want 0)" | tee -a "$exec_out"
[ "$with_flag_rc" -ne 0 ] || { echo "FAIL: flag did not reject UB"; exit 1; }
[ "$without_flag_rc" -eq 0 ] || { echo "FAIL: broken code should warn-only without flag"; exit 1; }

echo "--- 3. evidence: miss test on the no-flag broken build ---" | tee -a "$exec_out"
cp "$SCRATCH/broken_ok.o" src/fakegpu_pool.o
nice -n 10 make > /dev/null 2>&1
nice -n 10 gcc -O2 -g -Iinclude -Ishim -o build/test_resolver_attach \
    tests/test_resolver_attach.c -L. -lfakegpu -Wl,-rpath,"$PWD" || exit 1
./build/test_resolver_attach > "$SCRATCH/attach_broken.out" 2>&1
attach_rc=$?
grep -E "miss returns|attach gate|FAILED" "$SCRATCH/attach_broken.out" | tee -a "$exec_out"
echo "attach-on-broken rc=$attach_rc (evidence only: rax luck, not gated)" | tee -a "$exec_out"
rm -f src/*.o && nice -n 10 make > /dev/null 2>&1  # restore fixed lib

OUT="${XY8_OUTDIR:-$SCRATCH}"
CUDA_VISIBLE_DEVICES= nice -n 10 python3 - "$OUT/metrics.json" <<EOF
import json
m = {
    "bead": "gpu-sim-xy8",
    "branch": "lab/gpu-sim-xy8",
    "clean_rebuild_warnings": $warns,
    "flag_rejects_broken": True,
    "noflag_builds_broken": True,
    "miss_test_note": ("attach gate on the no-flag broken build OBSERVED "
                         "exit 0 (rax luck) in the producing run; UB codegen "
                         "is not deterministic, so the rc is evidence in "
                         "output.log, not a gated metric"),
}
json.dump(m, open("$OUT/metrics.json", "w"), indent=2, sort_keys=True)
open("$OUT/metrics.json", "a").write("\n")
print("metrics written to $OUT")
EOF
cp "$SCRATCH/out.log" "$OUT/output.log"
if [ "$OUT" = "$SCRATCH" ]; then
    cmp -s "$HERE/metrics.json" "$SCRATCH/metrics.json" \
        && echo "RECOMPUTE PASS: 0 warnings; flag rejects UB; no-flag builds it" \
        || { echo "RECOMPUTE FAIL: metrics differ"; exit 1; }
fi