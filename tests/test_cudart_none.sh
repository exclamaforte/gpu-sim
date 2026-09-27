#!/usr/bin/env bash
# test_cudart_none.sh — mechanical acceptance test for gpu-sim-u2o.6.1.3.
#
# CPU-only: no GPU is touched, and CUDA_VISIBLE_DEVICES is blanked so a
# regression that reintroduces a real-CUDA dependency fails loudly instead of
# silently "passing" on the 5090.
#
# Asserts, in order:
#   1. nvcc -cudart=none links the driver against libfakegpu.
#   2. ldd matches neither libcudart nor libcuda.
#   3. the binary runs (exit 0) and prints the configured fake latency.
#   4. the same compile WITHOUT -lfakegpu FAILS to link -- i.e. libfakegpu is
#      the only provider of these symbols, so there is no silent fallthrough
#      to a real CUDA library.
#   5. the new symbols are present in nm -D --defined-only libfakegpu.so.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"

SRC=tests/test_cudart_none.cu
OUT=build/test_cudart_none
NEG=build/test_cudart_none_neg
LATENCY_MS=${FAKEGPU_TEST_LATENCY_MS:-2}
ARCH=${FAKEGPU_TEST_ARCH:-sm_120}
rc=0
mkdir -p build

fail() { echo "FAIL: $*"; rc=1; }
ok()   { echo "PASS: $*"; }

# ---- 1. link against the fake only --------------------------------------
if nvcc -cudart=none -arch="$ARCH" -O2 "$SRC" -o "$OUT" \
        -L. -lfakegpu -Xlinker -rpath -Xlinker "$REPO" 2>build/cc_none.log; then
    ok "links with -cudart=none -lfakegpu"
else
    fail "link with -cudart=none -lfakegpu"
    sed -n 1,15p build/cc_none.log
    exit 1
fi

# ---- 2. no real CUDA library is linked -----------------------------------
ldd_out="$(ldd "$OUT")"
if echo "$ldd_out" | grep -qiE 'libcudart|libcuda\.so'; then
    fail "binary links a real CUDA library:"; echo "$ldd_out"
else
    ok "ldd matches neither libcudart nor libcuda"
fi
if echo "$ldd_out" | grep -q libfakegpu.so; then
    ok "ldd shows libfakegpu as the provider"
else
    fail "libfakegpu not in ldd output"
fi

# ---- 3. it runs, GPU-free, and reports the configured latency ------------
run_out="$(CUDA_VISIBLE_DEVICES= FAKEGPU_KERNEL_LATENCY_MS="$LATENCY_MS" \
           timeout 120 "$OUT" 2>&1)"
run_rc=$?
if [ "$run_rc" -ne 0 ]; then
    fail "smoke run exited $run_rc"; echo "$run_out" | sed -n 1,10p
else
    ok "smoke run exits 0 with CUDA_VISIBLE_DEVICES= (no GPU contact)"
fi
n_trials=$(printf '%s\n' "$run_out" | grep -c '"trial"')
if [ "$n_trials" -eq 3 ]; then
    ok "printed 3 trial JSON lines"
else
    fail "expected 3 trial lines, got $n_trials"; echo "$run_out" | sed -n 1,10p
fi
if printf '%s\n' "$run_out" | grep -q "\"time_ms\": ${LATENCY_MS}\.0*"; then
    ok "reported the fake's configured latency (${LATENCY_MS} ms)"
else
    fail "configured latency ${LATENCY_MS} ms not reported"; echo "$run_out" | sed -n 1,10p
fi
if printf '%s\n' "$run_out" | grep -q '"bad_launch_rejected": true'; then
    ok "invalid launch config is rejected by the fake"
else
    fail "invalid launch config was NOT rejected"; echo "$run_out" | sed -n 1,10p
fi

# ---- 4. without the fake the same compile must fail -----------------------
if nvcc -cudart=none -arch="$ARCH" -O2 "$SRC" -o "$NEG" 2>build/cc_neg.log; then
    fail "compiled WITHOUT -lfakegpu: libfakegpu is not the only provider!"
    rm -f "$NEG"
else
    if grep -qE 'cudaMalloc|__cudaLaunchKernel|__cudaRegisterFatBinary' build/cc_neg.log; then
        ok "same compile without -lfakegpu fails to link (no silent fallthrough)"
    else
        fail "link failed for an unexpected reason"; sed -n 1,15p build/cc_neg.log
    fi
fi

# ---- 5. the symbols are really exported ----------------------------------
# NOTE: grep -q exits at the first match, which SIGPIPEs `nm`; with
# `set -o pipefail` that would make every check look like a failure. Capture
# the listings once and grep the files instead.
nm -D --defined-only libfakegpu.so > build/nm_exported.txt 2>/dev/null
missing=""
for sym in __cudaRegisterFatBinary __cudaRegisterFatBinaryEnd \
           __cudaRegisterFunction __cudaUnregisterFatBinary \
           __cudaPushCallConfiguration __cudaPopCallConfiguration \
           __cudaGetKernel __cudaLaunchKernel \
           cudaGetLastError cudaPeekAtLastError; do
    grep -qE " T $sym\$" build/nm_exported.txt || missing="$missing $sym"
done
if [ -n "$missing" ]; then
    fail "symbols not exported by libfakegpu.so:$missing"
else
    ok "all new symbols exported by libfakegpu.so"
fi

exit $rc
