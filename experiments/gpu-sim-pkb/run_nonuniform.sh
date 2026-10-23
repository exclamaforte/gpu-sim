#!/bin/bash
# run_nonuniform.sh -- non-uniform fill allreduce value check (gpu-sim-0i8).
# Wraps allreduce_test_nu with the same env/topology/plugin as run_selfdesc.sh.
set -u
S=${1:?usage: run_nonuniform.sh <S_bytes> [run_id] [proto]}
RUNID=${2:-nu_${S}}
PROTO=${3:-Simple}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M8="$HERE/../gpu-sim-m3d.8"
BIN="$M8/build/allreduce_test_nu"
PLUGIN="/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-eut/experiments/gpu-sim-m3d.8/build/libncclNetSimFab.so"

NCCL_INC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
CUDA_INC="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
CUDA_LIB="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib"

if [ ! -x "$BIN" ]; then
  mkdir -p "$M8/build"
  gcc -O2 -I"$NCCL_INC" -I"$CUDA_INC" "$M8/allreduce_test_nu.c" -o "$BIN" \
      -L"$NCCL_LIB" -L"$CUDA_LIB" -l:libnccl.so.2 -l:libcudart.so.13 -lm \
      -Wl,-rpath,"$NCCL_LIB" -Wl,-rpath,"$CUDA_LIB" || exit 1
fi
[ -f "$PLUGIN" ] || { echo "FAIL: sim fabric plugin not built at $PLUGIN"; exit 1; }

FLOATS=$((S / 4))
UIDF="/tmp/nonuniform-uid-$$"
T0="/tmp/${RUNID}.trace.rank0"
T1="/tmp/${RUNID}.trace.rank1"
L0="$HERE/traces/run_${RUNID}_rank0.txt"
L1="$HERE/traces/run_${RUNID}_rank1.txt"
mkdir -p "$HERE/traces"
rm -f "$UIDF" "$T0" "$T1"

export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GPUSIM_RUN_ID="$RUNID" GPUSIM_NRANKS=2
export GPUSIM_TIMEOUT_NS=5000000 GPUSIM_ATTACH_TIMEOUT_NS=5000000000 GPUSIM_FAB_DELAY_US=0
export GPUSIM_DTYPE=float32
export GPUSIM_TOPOLOGY="$HERE/topo_single_eut.bin"
export NCCL_NET_PLUGIN="$PLUGIN"
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
if [ -n "$PROTO" ] && [ "$PROTO" != "default" ] && [ "$PROTO" != "none" ]; then export NCCL_PROTO="$PROTO"; fi

timeout 120s env NCCL_HOSTID=node1 GPUSIM_RANK=1 GPUSIM_TRACE="$T1" \
    "$BIN" 1 2 "$UIDF" "$FLOATS" > "$L1" 2>&1 &
P1=$!
sleep 1
timeout 120s env NCCL_HOSTID=node0 GPUSIM_RANK=0 GPUSIM_TRACE="$T0" \
    "$BIN" 0 2 "$UIDF" "$FLOATS" > "$L0" 2>&1 &
P0=$!
wait $P0; R0=$?
wait $P1; R1=$?
cat "$T0" "$T1" > "$HERE/traces/trace_${RUNID}.log" 2>/dev/null
rm -f "$T0" "$T1" "$UIDF"

ISENDS=$(grep -c "v12.isend isend" "$HERE/traces/trace_${RUNID}.log" 2>/dev/null)
NETSIM=$(grep -c "via NET/sim" "$L0" 2>/dev/null)
echo "size=$S proto=${PROTO:-default} run=$RUNID exit0=$R0 exit1=$R1 isends=$ISENDS netsim=$NETSIM"
grep -h "VALUE\|MISMATCH" "$L0" "$L1" 2>/dev/null | sed 's/^/  /'

[ "$NETSIM" -gt 0 ] || { echo "FAIL: no 'via NET/sim' line"; exit 1; }
[ "$R0" -eq 0 ] && [ "$R1" -eq 0 ] || { echo "FAIL: a rank did not exit 0"; exit 1; }
grep -q "VALUE OK" "$L0" && grep -q "VALUE OK" "$L1" || { echo "FAIL: payload mismatch"; exit 1; }
echo "PASS: non-uniform payload exact on both ranks"