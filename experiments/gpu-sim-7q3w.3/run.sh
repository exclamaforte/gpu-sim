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
#   6  wire_probe: two real processes exchange (qpn,rkey,addr) like NCCL does
#   5  abi_probe: QP state machine, one WR -> exactly one timed completion,
#      empty poll = 0, RDMA WRITE/WRITE_WITH_IMM/READ byte movement,
#      comp-channel event, dmabuf counter, modelled latency and 2x bandwidth
#      and drive WRITE / WRITE_WITH_IMM / READ across a MAP_SHARED ring, with
#      the 20 ms latency arm timed on the sender's own completion
#
# The NCCL-against-this-shim arms (device named by NCCL, IB transport chosen,
# all-reduce numerics, topology injection) are NOT in this script yet: they
# need the synthetic sysfs (per-rank mount namespace).  See provenance.md.
set -euo pipefail

# One thread per numeric library (tests/test_pin_coverage.py): the arms here are
# single-threaded by construction, and an unpinned BLAS pool on this box competes
# with the two-CPU quota the whole experiment runs under.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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
# --version-script: NCCL resolves its verbs table with dlvsym(name, "IBVERBS_1.1")
# (ibvsymbols.cc), so an unversioned shim makes NCCL fall back to Socket with
# only a WARN.  That was the first GPU run of this bead; see ibverbs.map.
$CC -shared -fPIC -O2 -Wall -Wextra -I"$EXP/vendor" \
    -Wl,--version-script="$EXP/ibverbs.map" \
    -o "$BUILD/libibverbs.so.1" "$EXP/ibverbs_shim.c" -lpthread
ln -sfn libibverbs.so.1 "$BUILD/libibverbs.so"
echo "shim built: $(stat -c%s "$BUILD/libibverbs.so.1") bytes"

stage "4. dlopen by SONAME + dlsym/dlvsym NCCL's name lists"
$CC -O1 -Wall -Wextra -o "$BUILD/dlopen_check" "$EXP/dlopen_check.c" -ldl
LD_LIBRARY_PATH="$BUILD" "$BUILD/dlopen_check" libibverbs.so.1 || fail "dlsym surface"
# NCCL's own table: dlvsym(name, "IBVERBS_1.1") for 26 names, plus the
# cherry-picked versions.  A NULL here is the difference between NCCL using
# our IB transport and NCCL silently using Socket.
$CC -O1 -Wall -Wextra -o "$BUILD/dlvsym_probe" "$EXP/dlvsym_probe.c" -ldl
LD_LIBRARY_PATH="$BUILD" "$BUILD/dlvsym_probe" libibverbs.so.1 \
    > "$BUILD/dlvsym.log" 2>&1 || { cat "$BUILD/dlvsym.log"; fail "dlvsym surface"; }
cat "$BUILD/dlvsym.log"
# and the same through the UNVERSIONED soname, which is what NCCL dlopens first
LD_LIBRARY_PATH="$BUILD" "$BUILD/dlvsym_probe" libibverbs.so \
    >> "$BUILD/dlvsym.log" 2>&1 || { cat "$BUILD/dlvsym.log"; fail "dlvsym via libibverbs.so"; }

stage "5. abi_probe (semantics, timing, counters)"
$CC -O1 -Wall -Wextra -I"$EXP/vendor" -o "$BUILD/abi_probe" "$EXP/abi_probe.c" \
    -L"$BUILD" -libverbs -Wl,-rpath,"$BUILD"
"$BUILD/abi_probe" > "$BUILD/probe.log" 2>&1 || { cat "$BUILD/probe.log"; fail "abi_probe"; }
cat "$BUILD/probe.log"
grep -q '^PROBE OK$' "$BUILD/probe.log" || fail "probe did not print PROBE OK"
RATIO=$(sed -n 's/.*ratio=\([0-9.]*\).*/\1/p' "$BUILD/probe.log")
awk -v r="$RATIO" 'BEGIN { if (r < 1.9 || r > 2.1) { printf "ratio %s out of [1.9,2.1] (modelled charge)\n", r; exit 1 } }' \
    || fail "2x line-rate ratio"

stage "6. wire_probe: two real processes, one shim"
$CC -O1 -Wall -Wextra -I"$EXP/vendor" -o "$BUILD/wire_probe" "$EXP/wire_probe.c" \
    -L"$BUILD" -libverbs -Wl,-rpath,"$BUILD"
GPU_SIM_ROOT="$ROOT" timeout 60 "$BUILD/wire_probe" > "$BUILD/wire.log" 2>&1 \
    || { cat "$BUILD/wire.log"; fail "wire_probe"; }
cat "$BUILD/wire.log"
grep -q '^WIRE OK$' "$BUILD/wire.log" || fail "wire probe did not print WIRE OK"

stage "7. sysfs spoof: the ONE path NCCL's IB init reads"
# NCCL reads /sys/class/infiniband/<dev>/device via realpath (and nothing else
# on the IB path -- see sysfs_spoof.c).  /sys is read-only here, so the spoof
# is an LD_PRELOAD prefix rewrite into a tree we own.
$CC -O1 -Wall -Wextra -fPIC -shared -o "$BUILD/sysfs_spoof.so" "$EXP/sysfs_spoof.c" -ldl
$CC -O1 -Wall -Wextra -o "$BUILD/realpath_probe" "$EXP/realpath_probe.c"
rm -rf "$BUILD/sysfs"
for r in 0 1; do bash "$EXP/fake_sysfs.sh" "$r" "$BUILD/sysfs/rank$r"; done
LD_PRELOAD="$BUILD/sysfs_spoof.so" FAKEIB_SYSFS="$BUILD/sysfs/rank0" \
    "$BUILD/realpath_probe" > "$BUILD/realpath.log" 2>&1 \
    || { cat "$BUILD/realpath.log"; fail "realpath probe"; }
cat "$BUILD/realpath.log"
grep -q '^REALPATH OK$' "$BUILD/realpath.log" || fail "realpath probe did not print REALPATH OK"
# The two ranks must NOT report the same NIC bridge: NCCL merges ranks that
# share a PCI path into one node, which would delete the cross-node edge we
# are building.  The no-GPU fallback legitimately shares one bridge, so only
# enforce it when the host has a display/3D device.
if grep -q 0x030000 /sys/bus/pci/devices/*/class 2>/dev/null; then
    b0=$(realpath -e "$BUILD/sysfs/rank0/infiniband/mlx5_0/device")
    b1=$(realpath -e "$BUILD/sysfs/rank1/infiniband/mlx5_0/device")
    [ "$b0" != "$b1" ] || fail "both ranks point their NIC at $b0; NCCL would merge them into one node"
    echo "distinct NIC bridges: rank0 $b0 / rank1 $b1"
fi
echo
echo "SHIM SELF-CHECK OK (stages 1-7)"
echo "NCCL arms: CPU-only, not the GPU queue ->  bash $EXP/gpu_arm.sh"
echo "  arm (a) transport selection is measured there; numerics and the data path"
echo "  are blocked on the simulator executing NCCL's kernels (provenance sec. 9)."
