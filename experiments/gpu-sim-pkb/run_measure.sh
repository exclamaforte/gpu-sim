#!/usr/bin/env bash
# Run one 2-rank allreduce at size S (bytes) and capture the plugin trace.
# Usage: run_measure.sh <S_bytes> <run_id> <trace_out_path>
set -uo pipefail
S_BYTES="$1"; RUN_ID="$2"; TRACE_OUT="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M8="$HERE/../../experiments/gpu-sim-m3d.8"
# build the size-flexible test binary if missing
if [ ! -x "$M8/build/allreduce_test_size" ]; then
  NCCL_INC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
  CUDA_INC="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
  NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
  CUDA_LIB="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib"
  mkdir -p "$M8/build"
  gcc -O2 -I"$NCCL_INC" -I"$CUDA_INC" "$M8/allreduce_test_size.c" -o "$M8/build/allreduce_test_size" \
      -L"$NCCL_LIB" -L"$CUDA_LIB" -l:libnccl.so.2 -l:libcudart.so.13 \
      -Wl,-rpath,"$NCCL_LIB" -Wl,-rpath,"$CUDA_LIB"
fi
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$PATH"
FLOATS=$((S_BYTES / 4))
export GPUSIM_RUN_ID="$RUN_ID"
export GPUSIM_NRANKS=2
export GPUSIM_TIMEOUT_NS=5000000
export GPUSIM_ATTACH_TIMEOUT_NS=5000000000
export GPUSIM_FAB_DELAY_US=0
export NCCL_NET_PLUGIN="/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-eut/experiments/gpu-sim-m3d.8/build/libncclNetSimFab.so"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4
export GPUSIM_DTYPE="float32"
# single-link topo bw=10 lat=5 loss=0
TOPo="$HERE/topo_single_eut.bin"
if [ ! -f "$TOPo" ]; then
  python3 - "$TOPo" <<'PY'
import sys
sys.path.insert(0, "/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-eut/epic_cluster")
from topology import emit
spec = {
  "version": 1,
  "models": {"m": {"sm_count":80,"clock_mhz":1000,"mem_bytes":1073741824,"mem_bw_gbs":1000.0,"arch":"x","source":"t"}},
  "nodes": [{"name":"n0","devices":[{"id":0,"model":"m"}]},{"name":"n1","devices":[{"id":0,"model":"m"}]}],
  "links": [{"a":{"node":"n0","device":0},"b":{"node":"n1","device":0},"bw_gbps":10.0,"latency_us":5.0,"mtu":4096,"loss_ppm":0,"jitter_us":0.0,"shared_group":None,"kind":"ethernet"}],
  "ranks": [{"rank":0,"node":"n0","device":0},{"rank":1,"node":"n1","device":0}],
}
open(sys.argv[1],"wb").write(emit(spec))
PY
fi
export GPUSIM_TOPOLOGY="$TOPo"
UIDFILE="/tmp/eut-uid-$$"; rm -f "$UIDFILE" "$TRACE_OUT"
export GPUSIM_TRACE="$TRACE_OUT"
T0="${TRACE_OUT}.rank0"; T1="${TRACE_OUT}.rank1"
: > "$T0"; : > "$T1"
# PI fix (gpu-sim-eut): the WAITING rank is started first. With rank 0 first, a
# small allreduce can complete before rank 1 attaches -- the plugin does not block
# a rank that finishes early -- so 4KiB/64KiB/256KiB either hung (exit 124) or
# captured partial, rank-asymmetric traffic. Also: no committed path is written;
# per-run logs live under traces/ so the committed pkb run0.log/run1.log survive.
LOGDIR="$HERE/traces"; mkdir -p "$LOGDIR"
LOG1="$LOGDIR/run_${RUN_ID}_rank1.txt"; LOG0="$LOGDIR/run_${RUN_ID}_rank0.txt"
timeout 120s env NCCL_HOSTID="node1" GPUSIM_RANK=1 GPUSIM_TRACE="$T1" "$M8/build/allreduce_test_size" 1 2 "$UIDFILE" "$FLOATS" > "$LOG1" 2>&1 &
P1=$!
sleep 1
timeout 120s env NCCL_HOSTID="node0" GPUSIM_RANK=0 GPUSIM_TRACE="$T0" "$M8/build/allreduce_test_size" 0 2 "$UIDFILE" "$FLOATS" > "$LOG0" 2>&1 &
P0=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "size=$S_BYTES run=$RUN_ID exit0=$R0 exit1=$R1"
# combine traces
cat "$T0" "$T1" > "$TRACE_OUT"
rm -f "$T0" "$T1"
# embed metadata header into trace? append a header line
{ echo "# S_BYTES=$S_BYTES NRANKS=2 RUN_ID=$RUN_ID"; cat "$TRACE_OUT"; } > "${TRACE_OUT}.tmp" && mv "${TRACE_OUT}.tmp" "$TRACE_OUT"
grep -q "isend" "$TRACE_OUT" || { echo "NO isend in trace"; exit 1; }
echo "trace written: $TRACE_OUT ($(wc -l < "$TRACE_OUT") lines)"
