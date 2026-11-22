#!/usr/bin/env bash
# gpu_arm.sh -- the NCCL acceptance arms of gpu-sim-7q3w.3.
#
# CPU-BOUND WORK: run it directly, NOT through gpqueue.
#
#   timeout 900 bash experiments/gpu-sim-7q3w.3/gpu_arm.sh
#
# The first version of this script was a GPU-queue job because it took two real
# devices.  This box has ONE NVIDIA GPU, and the queue killed the job for
# underuse (0.2% GPU over 5 min) -- correctly: the work was host-side waiting
# for a rank that could not start.  Both ranks now get a device from the repo's
# fake CUDA layer (FAKEGPU_NUM_DEVICES=2), so there is no real GPU work left to
# schedule, and holding the queue's only GPU for a host-side simulation is
# exactly what that policy forbids.
#
# Everything below NCCL is ours: libibverbs.so.1 = the shim, /sys/class/
# infiniband = the LD_PRELOAD realpath rewrite, bytes = the MAP_SHARED rings.
# The two ranks run in separate UTS namespaces with different hostnames, because
# NCCL decides same-node vs cross-node from what each rank reports about itself
# and a shared hostname is the one thing that guarantees the intra-node path.
set -euo pipefail

ROOT=${GPU_SIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
EXP="$ROOT/experiments/gpu-sim-7q3w.3"
BUILD="$ROOT/.lab/tmp/build-7q3w3-gpu"
# libnccl lives where a previous experiment extracted it, or gets extracted
# again here.  The worktree is not the main tree, so 7q3w.1's copy is found via
# the git common dir, not by a relative path that only works from the main tree.
MAIN=$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")
LIBS="$MAIN/experiments/gpu-sim-7q3w.1/libs/cu13"
[[ -s "$LIBS/libnccl.so.2" ]] || LIBS="$ROOT/.lab/tmp/nccl-cu13"
CUDA=${CUDA_SYSROOT:-/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux}
NCCLINC=$(ls -d "$HOME"/.local/lib/python3.*/site-packages/nvidia/nccl/include 2>/dev/null | tail -1)
CC=${CC:-gcc}

fail() { echo "GPU ARM FAIL: $*"; exit 1; }
note() { echo "--- $*"; }

[[ -n "${NCCLINC:-}" && -f "$NCCLINC/nccl.h" ]] || fail "nccl.h not found (pip nvidia-nccl-cu13)"
[[ -f "$LIBS/libnccl.so.2" ]] || {
    note "extracting libnccl into $LIBS (one-time, ~200 MB)"
    mkdir -p "$LIBS" "$ROOT/.lab/tmp/wheels"
    exec 9>"$LIBS/.extract.lock"
    flock -w 300 9 || fail "extraction lock timeout"
    WHEELDIR="$ROOT/.lab/tmp/wheels"
    ls "$WHEELDIR"/nvidia_nccl_cu13-*.whl >/dev/null 2>&1 \
        || pip download nvidia-nccl-cu13 --no-deps -d "$WHEELDIR" >/dev/null 2>&1 \
        || fail "pip download nvidia-nccl-cu13"
    WHL=$(ls "$WHEELDIR"/nvidia_nccl_cu13-*.whl | head -1)
    ( cd "$LIBS" && unzip -o "$WHL" "nvidia/nccl/lib/libnccl.so*" >/dev/null 2>&1 \
        && cp nvidia/nccl/lib/libnccl.so.2 libnccl.so.2 ) || fail "unzip libnccl"
    ln -sf libnccl.so.2 "$LIBS/libnccl.so"
}
[[ -d "$CUDA/include" ]] || fail "CUDA sysroot missing at $CUDA"
command -v unshare >/dev/null || fail "unshare not installed"

mkdir -p "$BUILD"
RUN="$BUILD/run-$(date +%s)"
mkdir -p "$RUN"
note "run dir $RUN"

# ---------------------------------------------------------------- build ----
# --version-script is not optional: NCCL resolves its verbs with
# dlvsym(name, "IBVERBS_1.1"), and an unversioned shim makes it fall back to
# Socket.  run.sh stage 4 proves the versioning; this build must use it too, or
# the arm measures a shim NCCL never sees (it did once: the log named this
# directory's libibverbs.so as the library that failed dlvsym).
$CC -O2 -Wall -Wextra -I"$EXP/vendor" -Wl,--version-script="$EXP/ibverbs.map" \
    -o "$RUN/libibverbs.so.1" "$EXP/ibverbs_shim.c" -shared -fPIC -ldl -lpthread \
    || fail "shim build"
ln -sfn libibverbs.so.1 "$RUN/libibverbs.so"
$CC -O1 -Wall -Wextra -fPIC -shared -o "$RUN/sysfs_spoof.so" "$EXP/sysfs_spoof.c" -ldl || fail "spoof build"
$CC -O1 -Wall -Wextra -o "$RUN/nccl_arm" "$EXP/nccl_arm.c" \
    -I"$NCCLINC" -I"$CUDA/include" -L"$LIBS" -L"$CUDA/lib" \
    -lnccl -lcudart -ldl -Wl,-rpath,"$LIBS":"$CUDA/lib" || fail "arm build"

# ------------------------------------------------------------- topology ----
for r in 0 1; do bash "$EXP/fake_sysfs.sh" "$r" "$RUN/sysfs/rank$r"; done
b0=$(realpath -e "$RUN/sysfs/rank0/infiniband/mlx5_0/device")
b1=$(realpath -e "$RUN/sysfs/rank1/infiniband/mlx5_0/device")
[[ "$b0" != "$b1" ]] || fail "both ranks report NIC bridge $b0; NCCL would merge them into one node"
note "NIC bridges: rank0 $b0 / rank1 $b1"
[[ $(nproc) -ge 2 ]] || note "one CPU: the shim's poll loops will fight each other"

# The CUDA layer is the repo's simulator, not the real driver: one real GPU
# cannot give two ranks two devices, and the point of this bead is the layer
# BELOW NCCL.  libfakegpu.so comes from the main tree (a worktree has no build
# products), and check_libfakegpu_fresh.py owns its freshness on main.
FAKEGPU="$MAIN/libfakegpu.so"
[[ -f "$FAKEGPU" ]] || fail "libfakegpu.so missing at $FAKEGPU (build it on main: make libfakegpu)"
# NOT preloaded: libfakegpu.so also exports ncclAllReduce/ncclCommInitRank/...
# (the simulator has its own collective implementation).  As LD_PRELOAD it
# preempts real NCCL's symbols and the arm silently measures the simulator:
# that is exactly what run-1795349638 did -- "allreduce" completed with 1048576
# wrong elements, no NCCL_DEBUG output at all, and no bytes on our wire.  A
# version script cannot fix it after the fact (objcopy --localize-symbol does
# not touch .dynsym), so it is installed as the CUDA *library* instead:
# libcudart.so.13 / libcuda.so.1 symlinks in a directory that comes first.
# ld.so then loads it to satisfy libnccl's DT_NEEDED, and libnccl itself is
# earlier in the global scope than its own dependency, so NCCL's collectives
# stay NCCL's.  The arm's assertions check the difference: NCCL_DEBUG output
# must exist and bytes must move on our wire.
FAKECUDA="$RUN/fakecuda"
mkdir -p "$FAKECUDA"
# Only the RUNTIME is faked.  libcuda.so.1 stays the real driver: NCCL's driver
# wrapper checks cuDriverGetVersion (libfakegpu does not export it) and reported
# "Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'",
# which stopped ncclCommInitRank dead.  The real driver answering cu* calls about
# fake addresses costs GDR/dmabuf (arm (c)) -- measured separately rather than
# pretended.
for so in libcudart.so libcudart.so.13; do
    ln -sfn "$FAKEGPU" "$FAKECUDA/$so"
done
export LD_LIBRARY_PATH="$FAKECUDA:$RUN:$LIBS:$CUDA/lib"
export LD_PRELOAD="$RUN/sysfs_spoof.so"
export FAKEGPU_NUM_DEVICES=2          # two ranks, two virtual devices, one real GPU
export FAKEGPU_TOPO=                 # no synthetic topology file: default graph
export FAKEIB_RING_DIR="$RUN/rings"
export FAKEIB_RING_MB=64
export FAKEIB_NRANKS=2
export FAKEIB_BW_GBPS=400
export FAKEIB_LATENCY_US=1.2
export FAKEIB_TRACE="${ARM_TRACE:-}"   # ARM_TRACE=1 names the verbs call that hangs
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TRANSPORT
export NCCL_IB_HCA=mlx5_0
# NCCL 2.32 initializes two optional subsystems that crash on the fake device
# before any collective: ncclGinProxyInit (GPU-initiated networking) and the
# RMA/IB proxy plugin.  NCCL logs "Assigned plugin RMA_IB_PROXY" immediately
# before the segfault (coredumpctl: ncclGinProxyInit, ncclIbAsyncThreadMain).
# They serve paths this experiment does not model; enabling them is separate
# work, and their absence stays visible because the arm asserts the transport.
# Overridable so the next seam can be measured without editing this file:
# NCCL_RMA_DISABLE=0 runs NCCL's RMA/IB proxy, which is what posts the data WRs
# in 2.32 (with it disabled, channels are assigned to NET/IB but the shim's
# post_send counter never moves -- see provenance sec. 9).
export NCCL_GIN_ENABLE=${NCCL_GIN_ENABLE-0}
export NCCL_RMA_DISABLE=${NCCL_RMA_DISABLE-1}
export NCCL_ARM_DIR="$RUN/hs"
mkdir -p "$NCCL_ARM_DIR" "$FAKEIB_RING_DIR"

rank_log() { echo "$RUN/rank$1.log"; }

# Each rank: its own UTS hostname, its own virtual CUDA device, its own sysfs
# tree.  The CUDA device comes from the fake layer (FAKEGPU_NUM_DEVICES=2), not
# from CUDA_VISIBLE_DEVICES: this box has ONE NVIDIA GPU, so remapping a real
# index 1 is what made rank1 log 'no CUDA-capable device is detected' in job
# 20261122t113451-e1f979af while rank0 waited for it.  With the fake layer the
# device ordinal IS the logical GPU, and its PCI bus id is distinct per ordinal
# (cudaDeviceGetPCIBusId), which is what NCCL's topology needs.
run_rank() {
    local r=$1
    ARM_CUDA_DEV=$r \
    FAKEIB_RANK=$r \
    FAKEIB_SYSFS="$RUN/sysfs/rank$r" \
    FAKEIB_COUNTERS_FILE="$RUN/counters-rank$r.json" \
        unshare -u -r -- sh -c "hostname nccl-rank$r; exec '$RUN/nccl_arm' $r 2 ${ARM_ARGS:-}" \
        >"$(rank_log "$r")" 2>&1
}

note "launching rank1 then rank0 (each in its own UTS namespace)"
( sleep 0.3; run_rank 1; echo "rank1 rc=$?" > "$RUN/rank1.rc" ) &
R1=$!
run_rank 0 || echo "rank0 rc=$?" > "$RUN/rank0.rc"
wait $R1 || true

note "--- rank0 (last 40 lines) ---"; tail -40 "$(rank_log 0)"
note "--- rank1 (last 40 lines) ---"; tail -40 "$(rank_log 1)"

# ------------------------------------------------------------ assertions ----
failures=0
chk() { if ! eval "$1"; then echo "ASSERT FAIL: $2"; failures=$((failures+1)); else echo "ASSERT ok: $2"; fi; }

for r in 0 1; do
    L=$(rank_log "$r")
    # bad=0, not just "the line exists": the arm prints its allreduce line
    # either way, so a bare grep passed on 1048576 wrong elements (it did).
    chk "grep -q 'ARM ok allreduce .* bad=0$' '$L'" "rank$r all-reduce numerics (bad=0)"
    chk "grep -q 'ARM OK rank=$r' '$L'" "rank$r exit line"
    # (a) NCCL named OUR device and chose IB, with no plugin in the picture
    chk "grep -q 'NET/IB : Using \[0\]mlx5_0' '$L'" "rank$r: NCCL_DEBUG names our mlx5_0 as an IB device"
    chk "grep -q 'via NET/IB' '$L'" "rank$r: channel went via NET/IB, not P2P/SHM"
    # "Could not find: libnccl-net.so" is NCCL's normal "no plugin installed"
    # line, so the assertion must be about what was LOADED: Socket being
    # initialized or assigned is the failure this arm exists to catch.
    chk "! grep -q 'Initialized NET plugin Socket' '$L'" "rank$r: no net plugin was loaded (it must be NCCL's own IB)"
    chk "! grep -q 'Assigned NET plugin Socket' '$L'" "rank$r: Socket was never assigned to the comm"
    chk "! grep -q 'Could not find real path' '$L'" "rank$r: the sysfs spoof answered NCCL's realpath"
done
# Transport-level evidence: did bytes actually cross our QPs?  post_send_calls
# and bytes_written are the shim's own counters, not NCCL's opinion.
for r in 0 1; do
    C="$RUN/counters-rank$r.json"
    chk "[[ -s '$C' ]]" "rank$r counters dumped"
    chk "python3 -c 'import json,sys; d=json.load(open(\"$C\")); sys.exit(0 if d[\"bytes_written\"]>0 and d[\"post_send_calls\"]>0 else 1)'" \
        "rank$r: NCCL posted WRs and moved bytes on our QPs"
    # Arm (c) is NOT reachable on this box: the real libcuda cannot export
    # dma-bufs for memory the fake cudart allocated, so NCCL never gets a real
    # fd.  The honest assertion is therefore that the counter stays 0 -- it now
    # counts only real registrations (fd >= 0), so a 1 here would mean something
    # we did not model started working, and that must be looked at, not ignored.
    chk "python3 -c 'import json,sys; d=json.load(open(\"$C\")); sys.exit(0 if d[\"dmabuf_registrations\"]==0 else 1)'" \
        "rank$r: dmabuf counter is 0 (arm (c) needs a driver that can export fake memory)"
done
chk "[[ ! -e '$RUN/rank0.rc' ]]" "rank0 exited 0"
chk "[[ ! -e '$RUN/rank1.rc' ]]" "rank1 exited 0"

echo
if (( failures )); then
    echo "GPU ARM FAILED: $failures assertion(s) -- run dir kept at $RUN"
    exit 1
fi
echo "GPU ARM OK: NCCL's own IB transport ran a real all-reduce on the fake fabric"
# Keep the run dir on failure; on success only the summary is interesting.
rm -rf "$RUN/rings"
