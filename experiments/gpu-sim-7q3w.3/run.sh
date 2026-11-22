#!/usr/bin/env bash
# gpu-sim-7q3w.3: libibverbs.so.1 shim -- reproducible driver for everything
# that does NOT need NCCL yet.
#
# Stages (each fails the run):
#   1  vendored headers verify against their recorded SHA256
#   2  ABI dump: the compiled-against header vs the rdma-core counter-example,
#      diffed against the committed pin abi_diff_expected.txt
#   3  build the shim; the shim's _Static_asserts are part of the build
#   4  dlopen("libibverbs.so.1") + dlsym the names NCCL dlsyms
#   5  abi_probe: QP state machine, one WR -> exactly one timed completion,
#      empty poll = 0, RDMA WRITE/WRITE_WITH_IMM/READ byte movement,
#      comp-channel event, dmabuf counter, modelled latency and 2x bandwidth
#
# The NCCL-against-this-shim arms (device named by NCCL, IB transport chosen,
# all-reduce numerics, topology injection) are NOT in this script yet: they
# need the synthetic sysfs and the cross-process wire.  See provenance.md.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
ROOT="$(pwd)"
EXP="$ROOT/experiments/gpu-sim-7q3w.3"
BUILD="$ROOT/.lab/tmp/7q3w3-build-$$"
mkdir -p "$BUILD"
trap 'rm -rf "$BUILD"' EXIT
CC="${CC:-gcc}"

fail() { echo "STAGE FAIL: $*" >&2; exit 1; }
stage() { echo; echo "=== $* ==="; }

stage "1. vendored headers: SHA256"
(cd "$EXP/vendor" && sha256sum -c SHA256SUMS) || fail "vendor/infiniband SHA256"
(cd "$EXP/vendor/nccl" && sha256sum -c SHA256SUMS) || fail "vendor/nccl SHA256"
echo "vendored headers OK (rdma-core v59.3 reference; NCCL v2.32.6-1 ABI)"

stage "2. ABI dump: compiled-against vs reference"
$CC -O0 -I"$EXP/vendor" -DIBV_HEADER='"infiniband/verbs.h"' \
    -o "$BUILD/abi_vendor" "$EXP/abi_layout.c" 2>/dev/null
$CC -O0 -I"$EXP/vendor" -DIBV_HEADER='"nccl/ibvcore.h"' \
    -o "$BUILD/abi_nccl" "$EXP/abi_layout.c" 2>/dev/null
"$BUILD/abi_vendor" > "$BUILD/vendor.txt"
"$BUILD/abi_nccl" > "$BUILD/nccl.txt"
# The pin: exactly these three differences, measured on 2026-11-22, and no
# others.  If a header swap changes a field we touch, this diff changes and
# the stage fails instead of letting a wrong offset reach NCCL.
diff "$BUILD/vendor.txt" "$BUILD/nccl.txt" > "$BUILD/abi.diff" || true
diff -u "$EXP/abi_diff_expected.txt" "$BUILD/abi.diff" || \
    fail "ABI diff moved; re-derive abi_diff_expected.txt before trusting the shim"
echo "ABI pin OK: 3 size differences, every field the shim touches agrees"

stage "3. build shim (its _Static_asserts pin the NCCL ABI at compile time)"
$CC -shared -fPIC -O2 -Wall -Wextra -I"$EXP/vendor" \
    -o "$BUILD/libibverbs.so.1" "$EXP/ibverbs_shim.c" -lpthread
ln -sfn libibverbs.so.1 "$BUILD/libibverbs.so"
echo "shim built: $(stat -c%s "$BUILD/libibverbs.so.1") bytes"

stage "4. dlopen by SONAME + dlsym NCCL's name list"
$CC -O1 -Wall -Wextra -o "$BUILD/dlopen_check" "$EXP/dlopen_check.c" -ldl
LD_LIBRARY_PATH="$BUILD" "$BUILD/dlopen_check" libibverbs.so.1 || fail "dlsym surface"

stage "5. abi_probe (semantics, timing, counters)"
$CC -O1 -Wall -Wextra -I"$EXP/vendor" -o "$BUILD/abi_probe" "$EXP/abi_probe.c" \
    -L"$BUILD" -libverbs -Wl,-rpath,"$BUILD"
"$BUILD/abi_probe" > "$BUILD/probe.log" 2>&1 || { cat "$BUILD/probe.log"; fail "abi_probe"; }
cat "$BUILD/probe.log"
grep -q '^PROBE OK$' "$BUILD/probe.log" || fail "probe did not print PROBE OK"
RATIO=$(sed -n 's/.*ratio=\([0-9.]*\).*/\1/p' "$BUILD/probe.log")
awk -v r="$RATIO" 'BEGIN { if (r < 1.6 || r > 2.4) { printf "ratio %s out of [1.6,2.4]\n", r; exit 1 } }' \
    || fail "2x line-rate ratio"
echo
echo "SHIM SELF-CHECK OK (stages 1-5; NCCL arms pending: sysfs + wire)"
