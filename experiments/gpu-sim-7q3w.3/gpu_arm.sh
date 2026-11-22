#!/usr/bin/env bash
# gpu_arm.sh -- the NCCL acceptance arms of gpu-sim-7q3w.3.
#
# GPU-QUEUED ONLY. Two ranks need a device each, so this must run under
# scripts/gpu_run.sh (7q3w.1 precedent), never bare from a CPU-only shell:
#
#   scripts/gpu_run.sh 900s --log experiments/gpu-sim-7q3w.3/gpu_arm.log \
#       -- bash experiments/gpu-sim-7q3w.3/gpu_arm.sh
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
$CC -O2 -Wall -Wextra -I"$EXP/vendor" -o "$RUN/libibverbs.so.1" "$EXP/ibverbs_shim.c" -shared -fPIC -ldl -lpthread \
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

export LD_LIBRARY_PATH="$RUN:$LIBS:$CUDA/lib"
export LD_PRELOAD="$RUN/sysfs_spoof.so"
export FAKEIB_RING_DIR="$RUN/rings"
export FAKEIB_RING_MB=64
export FAKEIB_NRANKS=2
export FAKEIB_BW_GBPS=400
export FAKEIB_LATENCY_US=1.2
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TRANSPORT
export NCCL_IB_HCA=mlx5_0
export NCCL_ARM_DIR="$RUN/hs"
mkdir -p "$NCCL_ARM_DIR" "$FAKEIB_RING_DIR"

rank_log() { echo "$RUN/rank$1.log"; }

# Each rank: its own UTS hostname, its own CUDA device, its own sysfs tree.
run_rank() {
    local r=$1
    CUDA_VISIBLE_DEVICES=$r \
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
    chk "grep -q 'ARM ok allreduce' '$L'" "rank$r all-reduce numerics line"
    chk "grep -q 'ARM OK rank=$r' '$L'" "rank$r exit line"
    # (a) NCCL named OUR device and chose IB, with no plugin in the picture
    chk "grep -q 'NET/IB : Using \[0\]mlx5_0' '$L'" "rank$r: NCCL_DEBUG names our mlx5_0 as an IB device"
    chk "grep -q 'via NET/IB' '$L'" "rank$r: channel went via NET/IB, not P2P/SHM"
    chk "! grep -qi 'net plugin\\|NET/Plugin' '$L'" "rank$r: no net plugin was loaded (it must be NCCL's own IB)"
    chk "! grep -q 'Could not find real path' '$L'" "rank$r: the sysfs spoof answered NCCL's realpath"
done
# (c) the shim saw dmabuf registrations and moved bytes / (b) is stage 5
for r in 0 1; do
    C="$RUN/counters-rank$r.json"
    chk "[[ -s '$C' ]]" "rank$r counters dumped"
    chk "python3 -c 'import json,sys; d=json.load(open(\"$C\")); sys.exit(0 if d[\"dmabuf_registrations\"]>0 and d[\"bytes_written\"]>0 else 1)'" \
        "rank$r: dmabuf registrations > 0 and bytes_written > 0"
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
