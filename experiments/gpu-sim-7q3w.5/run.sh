#!/usr/bin/env bash
# gpu-sim-7q3w.5 first experiment: 2 ranks, 1 MiB float32 allreduce through
# GENUINE NCCL on the modelled IB fabric, with the 7q3w.4 aliased-device shims
# providing one virtual GPU per rank on this box's one real GPU.
#
# Run through the GPU queue:
#   ts --max-runtime 10m -L '7q3w.5 2-rank NCCL numerics' -- \
#       bash experiments/gpu-sim-7q3w.5/run.sh
#
# Produces a run dir under .lab/tmp and prints machine-checkable ASSERT lines.
# The assertion is EXACT: bad=0 and max_abs_err=0.0, not a tolerance.
set -euo pipefail

ROOT=${GPU_SIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
EXP5="$ROOT/experiments/gpu-sim-7q3w.5"
EXP3="$ROOT/experiments/gpu-sim-7q3w.3"
EXP4="$ROOT/experiments/gpu-sim-7q3w.4"
MAIN=$(dirname "$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir)")
LIBS="$MAIN/experiments/gpu-sim-7q3w.1/libs/cu13"
CUDA=${CUDA_SYSROOT:-/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux}
NCCLINC=$(ls -d "$HOME"/.local/lib/python3.*/site-packages/nvidia/nccl/include 2>/dev/null | tail -1)
CC=${CC:-gcc}
N=${N:-262144}                    # 1 MiB float32
BUILD="$ROOT/.lab/tmp/build-7q3w5-numerics"
RUN="$BUILD/run-$(date +%Y%m%dT%H%M%S)"

fail() { echo "NUMERICS RUN FAIL: $*"; exit 1; }
note() { echo "--- $*"; }

[[ -f "$LIBS/libnccl.so.2" ]] || fail "libnccl.so.2 missing at $LIBS"
[[ -n "${NCCLINC:-}" && -f "$NCCLINC/nccl.h" ]] || fail "nccl.h not found"
[[ -d "$CUDA/include" ]] || fail "CUDA sysroot missing at $CUDA"
command -v unshare >/dev/null || fail "unshare not installed"
mkdir -p "$BUILD" "$RUN"
note "run dir $RUN"

# ------------------------------------------------------------- build ----
$CC -O2 -Wall -Wextra -I"$EXP3/vendor" -Wl,--version-script="$EXP3/ibverbs.map" \
    -Dibv_query_ece=fakeib_query_ece_orig -Dibv_set_ece=fakeib_set_ece_orig \
    -o "$RUN/libibverbs.so.1" "$EXP3/ibverbs_shim.c" "$EXP5/ece_shim.c" \
    -shared -fPIC -ldl -lpthread || fail "ibverbs shim build"
ln -sfn libibverbs.so.1 "$RUN/libibverbs.so"
$CC -O1 -Wall -Wextra -fPIC -shared -o "$RUN/sysfs_spoof.so" "$EXP3/sysfs_spoof.c" -ldl \
    || fail "sysfs spoof build"
$CC -O1 -Wall -Wextra -o "$RUN/nccl_numerics" "$EXP5/nccl_numerics.c" \
    -I"$NCCLINC" -I"$CUDA/include" -L"$LIBS" -L"$CUDA/lib" \
    -lnccl -lcudart -ldl -Wl,-rpath,"$LIBS":"$CUDA/lib" || fail "numerics build"

# The 7q3w.4 alias shims: build once, then make one variant per rank with a
# one-device count and the REAL PCI bus of that rank's display GPU.  NCCL's
# topology needs the selected GPU bus to sit under the same PCI bridge as the
# fake NIC, so the synthetic D1 bus base cannot be used for this arm.
note "building aliased-device shims"
python3 - "$EXP4" "$BUILD" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import probe_vdev
probe_vdev.build()
PY
VDEV_TEMPLATE="$ROOT/.lab/tmp/vdev-build"
[[ -f "$VDEV_TEMPLATE/libcuda-vdev.so.1" ]] || fail "vdev build template missing"

build_vdev_variant() {
    local r=$1 base=$2 out="$BUILD/vdev-r$1"
    rm -rf "$out"; cp -a "$VDEV_TEMPLATE" "$out"
    sed -i -E "s/#define B300_NVDEV [0-9]+/#define B300_NVDEV 1/; s/#define B300_PCI_BUS_BASE [0-9]+/#define B300_PCI_BUS_BASE $base/" "$out/b300_spec.h"
    $CC -shared -fPIC -O1 -I"$CUDA/include" -I"$out" -Wl,-soname,libcuda-vdev.so.1 \
        -o "$out/libcuda-vdev.so.1" "$EXP4/shim/vdev.c" "$out/forwarders.c" "$out/forwarders.S" -ldl \
        || fail "vdev r$r build"
    # Do NOT install the vdev NVML shim in this arm: NCCL calls
    # nvmlDeviceGetP2PStatus with a fake vdev handle, and the generated NVML
    # trampolines forward that handle to the REAL libnvidia-ml, which segfaults
    # (coredump 3114101: nvmlDeviceGetP2PStatus -> real NVML +0x40859).  This
    # experiment does not need an NVML identity for NCCL; the real NVML sees the
    # one real GPU, rank0's bus id has a handle, and rank1's synthetic bus id
    # simply has none.
    rm -f "$out/libnvidia-ml.so.1" "$out/libnvidia-ml.so"
    $CC -shared -fPIC -O1 -I"$CUDA/include" -I"$out" -Wl,-soname,libcudart-vdev.so.1 \
        -o "$out/libcudart-vdev.so.1" "$EXP4/shim/cudart.c" -ldl \
        || fail "cudart r$r build"
    ln -sfn libcuda-vdev.so.1 "$out/libcuda.so.1"
    grep -E '^#define B300_(NVDEV|PCI_BUS_BASE) ' "$out/b300_spec.h"
}
# rank0's display GPU is 0000:01:00.0; rank1's is 0000:7a:00.0 (the AMD iGPU,
# used only as a distinct PCI identity; both ranks compute on the real NVIDIA
# GPU behind the alias).
build_vdev_variant 0 1
build_vdev_variant 1 122

# ------------------------------------------------------------- topology ----
for r in 0 1; do bash "$EXP3/fake_sysfs.sh" "$r" "$RUN/sysfs/rank$r"; done
b0=$(realpath -e "$RUN/sysfs/rank0/infiniband/mlx5_0/device")
b1=$(realpath -e "$RUN/sysfs/rank1/infiniband/mlx5_0/device")
[[ "$b0" != "$b1" ]] || fail "both ranks report NIC bridge $b0; NCCL would merge them"
note "NIC bridges: rank0 $b0 / rank1 $b1"

export FAKEIB_TRACE="${ARM_TRACE:-}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,TRANSPORT
export NCCL_IB_HCA=mlx5_0
export NCCL_GIN_ENABLE=0
export NCCL_RMA_DISABLE=1
export NCCL_ARM_DIR="$RUN/hs"
export FAKEIB_RING_DIR="$RUN/rings"
export FAKEIB_RING_MB=64
export FAKEIB_NRANKS=2
export FAKEIB_BW_GBPS=400
export FAKEIB_LATENCY_US=1.2
# --- gpu-sim-7q3w.5 p2p arm -------------------------------------------------
# ARM_P2P=1 adds a 2-rank isend/irecv whose payload is asserted BYTE-IDENTICAL
# (mismatches=0), not a counter check.  ARM_SKIP_AR=1 drops the allreduce so the
# p2p path can be measured alone.
P2P=${ARM_P2P:-0}
P2P_BYTES=${P2P_BYTES:-4096}
SKIP_AR=${ARM_SKIP_AR:-0}
RANK_TIMEOUT=${RANK_TIMEOUT:-180}
mkdir -p "$NCCL_ARM_DIR" "$FAKEIB_RING_DIR"

run_rank() {
    local r=$1
    local V="$BUILD/vdev-r$r"
    ARM_CUDA_DEV=0 VDEV_NODE=$r VDEV_REAL_CUDA=/usr/lib/libcuda.so.1 \
    FAKEIB_RANK=$r FAKEIB_SYSFS="$RUN/sysfs/rank$r" \
    FAKEIB_COUNTERS_FILE="$RUN/counters-rank$r.json" \
    NUMERICS_P2P="$P2P" NUMERICS_P2P_BYTES="$P2P_BYTES" \
    NUMERICS_SKIP_ALLREDUCE="$SKIP_AR" \
    LD_LIBRARY_PATH="$V:$RUN:$LIBS:$CUDA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    LD_PRELOAD="$RUN/sysfs_spoof.so:$V/libcudart-vdev.so.1" \
    timeout "$RANK_TIMEOUT" unshare -u -r -- sh -c "hostname nccl-rank$r; exec '$RUN/nccl_numerics' $r 2 $N" \
        >"$RUN/rank$r.log" 2>&1
}

note "launching rank1 then rank0 (each in its own UTS namespace)"
( set +e; run_rank 1; echo $? > "$RUN/rank1.rc" ) &
R1=$!
sleep 0.5
set +e
run_rank 0
echo $? > "$RUN/rank0.rc"
wait $R1 || true
set -e

note "--- rank0 tail ---"; tail -60 "$RUN/rank0.log"
note "--- rank1 tail ---"; tail -60 "$RUN/rank1.log"
note "--- NCCL transport evidence (from each rank log) ---"
for r in 0 1; do
    grep -E 'NET/IB : Using|via NET/IB|Initialized NET plugin IB|Using network IB' \
        "$RUN/rank$r.log" | sed "s/^/rank$r: /" || true
done

failures=0
chk() { if eval "$1"; then echo "ASSERT ok: $2"; else echo "ASSERT FAIL: $2"; failures=$((failures+1)); fi; }

for r in 0 1; do
    L="$RUN/rank$r.log"
    if [[ "$SKIP_AR" != "1" ]]; then
        chk "grep -q 'NUMERICS allreduce .* bad=0$' '$L'" "rank$r: allreduce elements wrong = 0 (exact)"
        chk "grep -q 'NUMERICS allreduce .* max_abs_err=0.000e+00' '$L'" "rank$r: max_abs_err = 0.0 (exact)"
    fi
    chk "grep -q 'NUMERICS ok rank=$r' '$L'" "rank$r: process exit line"
    chk "grep -q 'NET/IB : Using \\[0\\]mlx5_0' '$L'" "rank$r: NCCL named our mlx5_0 as the IB device"
    chk "grep -q 'via NET/IB' '$L'" "rank$r: channel went via NET/IB, not P2P/SHM"
    chk "! grep -q 'Initialized NET plugin Socket' '$L'" "rank$r: no Socket fallback"
    C="$RUN/counters-rank$r.json"
    chk "[[ -s '$C' ]]" "rank$r: counters dumped"
    chk "python3 -c 'import json,sys; d=json.load(open(\"$C\")); sys.exit(0 if d[\"bytes_written\"]>0 and d[\"post_send_calls\"]>0 else 1)'" \
        "rank$r: NCCL posted WRs and moved bytes on our QPs"
done

if [[ "$P2P" == "1" ]]; then
    # The p2p claim is the RESULT: exact equality of the received bytes.
    chk "grep -q 'NUMERICS p2p rank=0 nbytes=$P2P_BYTES sent=1' '$RUN/rank0.log'" \
        "rank0: p2p $P2P_BYTES-byte payload handed to the wire"
    chk "grep -q 'NUMERICS p2p rank=1 nbytes=$P2P_BYTES mismatches=0$' '$RUN/rank1.log'" \
        "rank1: 0 of $P2P_BYTES received bytes differ from sent bytes (exact)"
    for r in 0 1; do
        C="$RUN/counters-rank$r.json"
        chk "python3 -c 'import json,sys; d=json.load(open(\"$C\")); sys.exit(0 if d[\"rnr_drops\"]==0 and d[\"remote_access_errors\"]==0 else 1)'" \
            "rank$r: no dropped recv WR and no remote access error on the wire"
    done
fi
chk "grep -qx 0 '$RUN/rank0.rc'" "rank0 exited 0"
chk "grep -qx 0 '$RUN/rank1.rc'" "rank1 exited 0"

echo
if (( failures )); then
    echo "NUMERICS RUN FAILED: $failures assertion(s); run dir kept at $RUN"
    exit 1
fi
if [[ "$P2P" == "1" && "$SKIP_AR" != "1" ]]; then
    echo "NUMERICS RUN OK: exact 2-rank allreduce AND byte-identical $P2P_BYTES-byte send/recv through genuine NCCL on the modelled IB wire"
elif [[ "$P2P" == "1" ]]; then
    echo "NUMERICS RUN OK: byte-identical $P2P_BYTES-byte send/recv through genuine NCCL on the modelled IB wire (allreduce arm skipped)"
else
    echo "NUMERICS RUN OK: exact 2-rank allreduce through genuine NCCL on the modelled IB wire"
fi
echo "RUN_DIR=$RUN"
rm -rf "$RUN/rings"
