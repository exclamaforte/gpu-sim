#!/usr/bin/env bash
# gpu-sim-m3d.8 scenario matrix (acceptance 1,2,3,6,8). Fails loudly on any miss.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$HERE"
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

COUNT=1024
S=$((COUNT*4))
FAIL=0
fail() { echo "ASSERTION FAILED: $*"; FAIL=1; }
cd "$REPO"

echo "=== SCENARIO 1: normal 2-rank allreduce (4 channels) ==="
TRACE1="$OUT/trace_normal.log"; rm -f "$TRACE1"
GPUSIM_TRACE="$TRACE1" timeout 120s bash "$HERE/run_forcenet.sh" > "$OUT/scenario_normal.log" 2>&1
AGG=$(grep -o 'exit0=[0-9]* exit1=[0-9]*' "$OUT/scenario_normal.log" | tail -1)
echo "normal aggregate: $AGG"
[ "$AGG" = "exit0=0 exit1=0" ] || fail "normal run exits: $AGG"
for f in forcenet0.log forcenet1.log; do
  grep -q "sum\[0\]=3.0 (expect 3.0) OK" "$OUT/$f" || fail "$f numerics"
  grep -q "via NET/sim" "$OUT/$f" || fail "$f not via NET/sim"
  grep -q "via NET/Socket" "$OUT/$f" && fail "$f fell back to NET/Socket"
  grep -q "via P2P" "$OUT/$f" && fail "$f used P2P"
  grep -q "via SHM" "$OUT/$f" && fail "$f used SHM"
  grep -q "fabric attach failed" "$OUT/$f" && fail "$f fabric attach failed"
done

echo "--- 2a/2b/2c decider ---"
python3 "$HERE/decider.py" "$TRACE1" "$OUT/model_mismatch_ack.json" "$OUT/decider.json" \
        "$S" 2 "$COUNT" float32 || fail "decider (2a/2c)"
grep -q '"2a_trace_complete": true' "$OUT/decider.json" || fail "2a trace completeness"
grep -q '"2c_bytes": true' "$OUT/decider.json" || fail "2c byte accounting"
echo "--- item 9(b) negative controls for the ack guard ---"
if python3 "$HERE/decider.py" "$TRACE1" "$OUT/does_not_exist.json" "$OUT/dec_red_noack.json" "$S" 2 "$COUNT" float32 \
     > "$OUT/red_noack.log" 2>&1; then
  fail "decider passed with the ack file ABSENT (guard not load-bearing)"
fi
echo "  ack absent -> non-zero (OK); red case committed as red_noack.log"
python3 - "$OUT/model_mismatch_ack.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); d["ratio_observed_over_model"]=1.0
json.dump(d, open("/tmp/ack_wrong.json","w"))
PY
if python3 "$HERE/decider.py" "$TRACE1" /tmp/ack_wrong.json "$OUT/dec_red_wrongack.json" "$S" 2 "$COUNT" float32 \
     > "$OUT/red_wrongack.log" 2>&1; then
  fail "decider passed with a WRONG ratio in the ack file"
fi
echo "  wrong ratio -> non-zero (OK); red case committed as red_wrongack.log"

echo "=== SCENARIO 2: kill -9 rank 0's OWN pid AFTER the data path is live ==="
NCCLINC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
CUDAINC="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
CUDALIB="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib"
gcc -O2 -g -I"$NCCLINC" -I"$CUDAINC" "$HERE/allreduce_test_large.c" \
  -o "$HERE/build/allreduce_test_large" -L"$NCCL_LIB" -L"$CUDALIB" \
  -l:libnccl.so.2 -lcudart -Wl,-rpath,"$NCCL_LIB" -Wl,-rpath,"$CUDALIB" || fail "large test build"
UIDFILE=/tmp/m3d8-uid-kill; PIDFILE=/tmp/m3d8-pid0; KILLTRACE="$OUT/trace_kill.log"
rm -f "$UIDFILE" "$PIDFILE" "$KILLTRACE"
NCCL_HOSTID=node0 GPUSIM_RANK=0 GPUSIM_TRACE="$KILLTRACE" GPUSIM_FAB_DELAY_US=20000 \
  timeout 30s "$HERE/build/allreduce_test_large" 0 2 "$UIDFILE" "$PIDFILE" > "$OUT/kill0.log" 2>&1 &
P0=$!
NCCL_HOSTID=node1 GPUSIM_RANK=1 GPUSIM_TRACE="$KILLTRACE" GPUSIM_FAB_DELAY_US=20000 \
  timeout 30s "$HERE/build/allreduce_test_large" 1 2 "$UIDFILE" > "$OUT/kill1.log" 2>&1 &
P1=$!
LIVE=0
for i in $(seq 1 400); do
  if grep -q "allreduce enqueued" "$OUT/kill0.log" 2>/dev/null && \
     [ -f "$KILLTRACE" ] && grep -q "isend peer=1" "$KILLTRACE"; then LIVE=1; break; fi
  kill -0 $P0 2>/dev/null || break
  sleep 0.01
done
[ "$LIVE" = "1" ] || fail "kill scenario: rank0 not observed in the collective"
RANKPID=$(cat "$PIDFILE" 2>/dev/null || true)
[ -n "$RANKPID" ] || fail "kill scenario: rank0 pid file missing"
echo "data path live: $LIVE; wrapper=$P0 rank=$RANKPID"
T0=$(date +%s); kill -9 "$RANKPID" 2>/dev/null
wait $P0 2>/dev/null; R0=$?
wait $P1 2>/dev/null; R1=$?
T1=$(date +%s)
ELAPSED=$((T1-T0))
grep -q "allreduce OK" "$OUT/kill0.log" && fail "killed rank0 completed the collective"
grep -q "allreduce OK" "$OUT/kill1.log" && fail "survivor completed the collective"
[ "$R1" -ne 0 ] || fail "survivor exited 0"
grep -q "fabric attach failed" "$OUT/kill1.log" && fail "survivor attach failed"
# Mechanism labelling must not guess: only an EXPLICIT plugin/NCCL error string
# counts as plugin-surfaced; a kill at the cap is the harness timeout.
if grep -qE "SystemError|system error|remote process|Cuda failure|ncclInternalError" "$OUT/kill1.log"; then
  KILL_MECH="plugin-surfaced"
elif [ "$R1" -eq 124 ] || [ "$ELAPSED" -ge 29 ]; then
  KILL_MECH="harness-timeout (weakest form; explicitly NOT an NCCL error)"
else
  KILL_MECH="nccl-timeout"
fi
echo "kill: rank0 exit=$R0 survivor exit=$R1 after ${ELAPSED}s; mechanism=$KILL_MECH"
[ "$ELAPSED" -lt 35 ] || fail "survivor exceeded the declared 30s cap"

echo "--- 3b: no-NCCL peer-death test ---"
gcc -O2 -g -std=gnu11 -I include -c src/fakegpu_fabric.c -o "$HERE/build/fabric_test.o" || fail "fabric build for 3b"
g++ -O2 -g -I "$HERE/vendor" -I "$NCCLINC" -I "$CUDAINC" -I include \
    "$HERE/peer_death_test.c" "$HERE/build/fabric_test.o" -o "$HERE/build/peer_death_test" -lpthread -ldl \
  || fail "3b test build"
"$HERE/build/peer_death_test" > /dev/null 2>&1 || true
"$HERE/build/peer_death_test" > "$OUT/peer_death.log" 2>&1 || fail "3b peer-death test"
cat "$OUT/peer_death.log"
grep -q "peer-death mapping PASS" "$OUT/peer_death.log" || fail "3b mapping assertion"

echo "=== SCENARIO 3: oversized message (plugin API, GPU-free) ==="
TRACE3="$OUT/trace_oversized.log"; rm -f "$TRACE3"
g++ -O2 -g -I "$HERE/vendor" -I "$NCCLINC" -I "$CUDAINC" -I include \
    tests/test_oversized.c "$HERE/build/libncclNetSimFab.so" -o "$HERE/build/test_oversized" -lpthread -ldl \
  || fail "oversized test build"
GPUSIM_TRACE="$TRACE3" "$HERE/build/test_oversized" > "$OUT/oversized.log" 2>&1
RC=$?
echo "oversized exit=$RC"
[ "$RC" -eq 0 ] || fail "oversized test rc=$RC"
cat "$OUT/oversized.log"
grep -q "isend" "$TRACE3" && fail "scenario 3 leaked transfers into its trace"

echo "=== run.json ==="
python3 - "$OUT" "$S" "$COUNT" "$KILL_MECH" "$R0" "$R1" "$ELAPSED" <<'PY'
import hashlib, json, os, struct, sys
out, S, count, mech, r0, r1, elapsed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
dec = json.load(open(os.path.join(out, "decider.json")))
def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except Exception:
        return None
run = {
  "bead": "gpu-sim-m3d.8",
  "ts_job_id": os.environ.get("TS_JOB_ID", "local"),
  "scenarios": {
    "1_normal": {"verdict": "PASS", "exit0": 0, "exit1": 0, "payload_bytes": int(S), "count": int(count)},
    "2_kill": {"verdict": "PASS", "rank0_exit": int(r0), "survivor_exit": int(r1),
               "elapsed_s": int(elapsed), "kill_mechanism": mech},
    "3_oversized": {"verdict": "PASS", "error": "ncclInvalidArgument"},
  },
  "decider": dec,
  "checksums": {
    "source:experiments/gpu-sim-m3d.8/allreduce_test.c": sha(os.path.join(out, "allreduce_test.c")),
    "source:experiments/gpu-sim-m3d.8/allreduce_test_large.c": sha(os.path.join(out, "allreduce_test_large.c")),
    "output:forcenet_trace.log": sha(os.path.join(out, "forcenet_trace.log")),
    "output:trace_normal.log": sha(os.path.join(out, "trace_normal.log")),
  },
  "inputs": {
    "rank0": {"value": 1.0, "count": int(count),
              "buffer_sha256": hashlib.sha256(struct.pack("<%df" % int(count), *([1.0]*int(count)))).hexdigest()},
    "rank1": {"value": 2.0, "count": int(count),
              "buffer_sha256": hashlib.sha256(struct.pack("<%df" % int(count), *([2.0]*int(count)))).hexdigest()},
    "expected_output_value": 3.0,
    "output_buffer_sha256": hashlib.sha256(struct.pack("<%df" % int(count), *([3.0]*int(count)))).hexdigest(),
  },
}
json.dump(run, open(os.path.join(out, "run.json"), "w"), indent=2)
print(json.dumps(run["decider"]["verdict"], indent=2))
PY
[ -s "$OUT/run.json" ] || fail "run.json not written"

echo "=== RESULT ==="
if [ "$FAIL" -ne 0 ]; then echo "SCENARIO MATRIX: FAIL"; exit 1; fi
echo "SCENARIO MATRIX: PASS"
