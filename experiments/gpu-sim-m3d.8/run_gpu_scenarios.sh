#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$PATH"
export GPUSIM_RUN_ID="gpu-scenarios-$$"
export GPUSIM_NRANKS=2
export GPUSIM_TIMEOUT_NS=5000000
export GPUSIM_FAB_DELAY_US=0
export NCCL_NET_PLUGIN="$HERE/build/libncclNetSimFab.so"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4
export GPUSIM_DTYPE="float32"

cd "$REPO"

echo "=== SCENARIO 1: normal 2-rank allreduce (4 channels) ==="
timeout 120s bash "$HERE/run_forcenet.sh" > "$HERE/scenario_normal.log" 2>&1
echo "normal exit aggregate: $(grep -o 'exit0=[0-9]* exit1=[0-9]*' "$HERE/scenario_normal.log" | tail -1)"
grep -i "via NET/sim" "$HERE/forcenet0.log" | head -2
grep -i "sum\[0\]=.*OK" "$HERE/forcenet0.log" "$HERE/forcenet1.log"

echo "=== SCENARIO 2: kill -9 one rank mid-allreduce ==="
# compile large test
NCCLINC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
CUDAINC="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
CUDALIB="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib"
gcc -O2 -g -I"$NCCLINC" -I"$CUDAINC" \
  "$HERE/allreduce_test_large.c" -o "$HERE/build/allreduce_test_large" \
  -L"$NCCL_LIB" -L"$CUDALIB" -l:libnccl.so.2 -lcudart -Wl,-rpath,"$NCCL_LIB" -Wl,-rpath,"$CUDALIB" 2>&1 | tail -3
UIDFILE="/tmp/m3d8-uid-kill"; rm -f "$UIDFILE"
NCCL_HOSTID="node0" GPUSIM_RANK=0 GPUSIM_FAB_DELAY_US=50000 timeout 30s "$HERE/build/allreduce_test_large" 0 2 "$UIDFILE" > "$HERE/kill0.log" 2>&1 &
P0=$!
NCCL_HOSTID="node1" GPUSIM_RANK=1 GPUSIM_FAB_DELAY_US=50000 timeout 30s "$HERE/build/allreduce_test_large" 1 2 "$UIDFILE" > "$HERE/kill1.log" 2>&1 &
P1=$!
sleep 1
# kill rank 0 mid-flight
kill -9 $P0 2>/dev/null
wait $P0 2>/dev/null; R0=$?
wait $P1 2>/dev/null; R1=$?
echo "kill scenario: rank0 (killed) exit=$R0, rank1 (survivor) exit=$R1"
echo "survivor exit=$R1 after hard-timeout path (declared cap 30s; rc 124 = timeout)"
echo "--- survivor tail ---"
tail -5 "$HERE/kill1.log"

echo "=== SCENARIO 3: oversized message (GPU-free) ==="
"$REPO/tests/test_oversized" > "$HERE/oversized.log" 2>&1
echo "oversized exit=$? (expect 0)"
cat "$HERE/oversized.log"
