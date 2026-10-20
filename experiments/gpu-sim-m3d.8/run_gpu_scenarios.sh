#!/usr/bin/env bash
# gpu-sim-m3d.8 scenario matrix. Fails loudly (non-zero) on any missed assertion,
# so a green run means a real NET/sim data path.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$PATH"
export GPUSIM_RUN_ID="gpu-scenarios-$$"
export GPUSIM_NRANKS=2
export GPUSIM_TIMEOUT_NS=5000000
export GPUSIM_ATTACH_TIMEOUT_NS=5000000000
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

FAIL=0
fail() { echo "ASSERTION FAILED: $*"; FAIL=1; }

cd "$REPO"

echo "=== SCENARIO 1: normal 2-rank allreduce (4 channels) ==="
rm -f "$HERE/forcenet_trace.log"
timeout 120s bash "$HERE/run_forcenet.sh" > "$HERE/scenario_normal.log" 2>&1
AGG=$(grep -o 'exit0=[0-9]* exit1=[0-9]*' "$HERE/scenario_normal.log" | tail -1)
echo "normal aggregate: $AGG"
[ "$AGG" = "exit0=0 exit1=0" ] || fail "normal run exits: $AGG"
grep -q "sum\[0\]=3.0 (expect 3.0) OK" "$HERE/forcenet0.log" || fail "rank0 numerics"
grep -q "sum\[0\]=3.0 (expect 3.0) OK" "$HERE/forcenet1.log" || fail "rank1 numerics"
grep -q "via NET/sim" "$HERE/forcenet0.log" || fail "rank0 not via NET/sim"
grep -q "via NET/sim" "$HERE/forcenet1.log" || fail "rank1 not via NET/sim"
grep -q "via NET/Socket" "$HERE/forcenet0.log" && fail "rank0 fell back to NET/Socket"
grep -q "via NET/Socket" "$HERE/forcenet1.log" && fail "rank1 fell back to NET/Socket"
grep -q "via P2P" "$HERE/forcenet0.log" && fail "rank0 used P2P"
grep -q "via SHM" "$HERE/forcenet0.log" && fail "rank0 used SHM"
grep -q "fabric attach failed" "$HERE/forcenet0.log" && fail "rank0 fabric attach failed"
grep -q "isend" "$HERE/forcenet_trace.log" || fail "trace has no isend (data path not live)"
grep -q "irecv" "$HERE/forcenet_trace.log" || fail "trace has no irecv (data path not live)"
grep -q "dtype=float32" "$HERE/forcenet_trace.log" || fail "trace missing dtype field"
echo "normal: $(grep -c 'via NET/sim' "$HERE/forcenet0.log") NET/sim channel lines rank0; $(wc -l < "$HERE/forcenet_trace.log") trace lines"

echo "=== SCENARIO 2: kill -9 rank 0 AFTER the plugin data path is live ==="
NCCLINC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
CUDAINC="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
CUDALIB="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib"
gcc -O2 -g -I"$NCCLINC" -I"$CUDAINC" \
  "$HERE/allreduce_test_large.c" -o "$HERE/build/allreduce_test_large" \
  -L"$NCCL_LIB" -L"$CUDALIB" -l:libnccl.so.2 -lcudart -Wl,-rpath,"$NCCL_LIB" -Wl,-rpath,"$CUDALIB" || fail "large test build"
UIDFILE="/tmp/m3d8-uid-kill"; rm -f "$UIDFILE"
KILLTRACE="$HERE/kill_trace.log"; rm -f "$KILLTRACE"
export GPUSIM_TRACE="$KILLTRACE"
export GPUSIM_FAB_DELAY_US=2000   # slow the mailbox so the data path stays live long enough to kill mid-flight
NCCL_HOSTID="node0" GPUSIM_RANK=0 timeout 30s "$HERE/build/allreduce_test_large" 0 2 "$UIDFILE" > "$HERE/kill0.log" 2>&1 &
P0=$!
NCCL_HOSTID="node1" GPUSIM_RANK=1 timeout 30s "$HERE/build/allreduce_test_large" 1 2 "$UIDFILE" > "$HERE/kill1.log" 2>&1 &
P1=$!
# wait until rank0's plugin is sending actual data (trace shows an isend), then kill
LIVE=0
for i in $(seq 1 200); do
  if [ -f "$KILLTRACE" ] && grep -q "isend" "$KILLTRACE"; then LIVE=1; break; fi
  if ! kill -0 $P0 2>/dev/null; then break; fi
  sleep 0.05
done
[ "$LIVE" = "1" ] || fail "kill scenario: never observed isend before kill (data path not live)"
echo "data path live before kill: $LIVE"
T0=$(date +%s)
kill -9 $P0 2>/dev/null
wait $P0 2>/dev/null; R0=$?
wait $P1 2>/dev/null; R1=$?
T1=$(date +%s)
echo "kill scenario: rank0 (killed) exit=$R0, rank1 (survivor) exit=$R1 after $((T1-T0))s (declared cap 30s)"
[ "$R0" -eq 137 ] || fail "killed rank0 exit=$R0 (expected 137)"
[ "$R1" -ne 0 ] || fail "survivor exited 0 -- peer death not surfaced"
if grep -q "fabric attach failed" "$HERE/kill1.log"; then fail "survivor fabric attach failed"; fi
grep -q "isend" "$KILLTRACE" || fail "kill trace has no isend"
if [ "$R1" -eq 124 ]; then
  echo "survivor path: hard timeout (rc 124) within declared 30s cap"
else
  echo "survivor path: bounded error exit rc=$R1"
fi
echo "--- survivor tail ---"
tail -n 5 "$HERE/kill1.log"

echo "=== SCENARIO 3: oversized message (GPU-free plugin API) ==="
"$REPO/tests/test_oversized" > "$HERE/oversized.log" 2>&1
RC=$?
echo "oversized exit=$RC (expect 0)"
[ "$RC" -eq 0 ] || fail "oversized test rc=$RC"
cat "$HERE/oversized.log"

echo "=== RESULT ==="
if [ "$FAIL" -ne 0 ]; then
  echo "SCENARIO MATRIX: FAIL"
  exit 1
fi
echo "SCENARIO MATRIX: PASS"
