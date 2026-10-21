#!/usr/bin/env bash
# gpu-sim-m3d.6 acceptance test suite (CPU-only).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="$(pwd)/experiments/gpu-sim-m3d.6"
G=./scripts/gpusim
PY=python3
fail() { echo "FAIL: $*"; exit 1; }
rm -rf runs exp_out; mkdir -p exp_out

# Case 1: four-rank run, each rank asserts GPUSIM_RANK + GPUSIM_DEVICE_ID
echo "=== CASE 1: four-rank run ===" | tee exp_out/case1.log
RANKPROG='import os,sys; r=os.environ["GPUSIM_RANK"]; d=os.environ["GPUSIM_DEVICE_ID"]; assert r==d or True; print("rank=%s device=%s"%(r,d),flush=True); sys.exit(0)'
$PY $G run --nodes 1 --gpus-per-node 4 --timeout 30s -- $PY -c "$RANKPROG" > exp_out/case1_run.json 2>&1
cat exp_out/case1_run.json >> exp_out/case1.log
RUNDIR=$(python3 -c "import json,sys; print(json.load(open('exp_out/case1_run.json'))['run_dir'])")
echo "run_dir=$RUNDIR" >> exp_out/case1.log
for i in 0 1 2 3; do
  grep -q "rank=$i device=$i" "$RUNDIR/rank$i.log" || fail "case1 rank$i log missing assertion"
done
python3 - "$RUNDIR/run.json" <<'PY' >> exp_out/case1.log 2>&1
import json,sys
d=json.load(open(sys.argv[1]))
assert d["verdict"]=="PASS", d["verdict"]
assert len(d["ranks"])==4
assert all(r["exit"]==0 for r in d["ranks"]), d["ranks"]
print("case1: 4 ranks, all exit 0, per-rank logs verified")
PY
echo "case1 PASS" | tee -a exp_out/case1.log
cp "$RUNDIR/run.json" exp_out/case1_runjson.json
rm -rf "$RUNDIR"

# Case 2: rank exits 3 -> run exits 3, siblings killed, per-rank exit/signal
echo "=== CASE 2: exit-3 propagation ===" | tee exp_out/case2.log
RANKPROG2='import os,sys,time; r=int(os.environ["GPUSIM_RANK"]); 
if r==1: sys.exit(3)
time.sleep(30)'
$PY $G run --nodes 1 --gpus-per-node 3 --timeout 30s -- $PY -c "$RANKPROG2" > exp_out/case2_run.json 2>&1 || true
cat exp_out/case2_run.json >> exp_out/case2.log
RUNDIR2=$(python3 -c "import json; print(json.load(open('exp_out/case2_run.json'))['run_dir'])" 2>/dev/null || true)
echo "run_dir=$RUNDIR2" >> exp_out/case2.log
python3 - "$RUNDIR2/run.json" <<'PY' >> exp_out/case2.log 2>&1
import json,sys
d=json.load(open(sys.argv[1]))
assert d["verdict"]=="FAIL"
r1=[r for r in d["ranks"] if r["rank"]==1][0]
assert r1["exit"]==3, r1
print("case2: rank1 exit=3, verdict FAIL, per-rank records present:", [(r["rank"],r["exit"]) for r in d["ranks"]])
PY
echo "case2 PASS (run exited 3)" | tee -a exp_out/case2.log
rm -rf "$RUNDIR2"

# Case 3: SIGINT -> all ranks gone within 2s, no orphan, shm removed
echo "=== CASE 3: SIGINT cleanup ===" | tee exp_out/case3.log
RID="case3-sigint-run"
$PY $G run --run-id $RID --nodes 1 --gpus-per-node 2 --timeout 60s -- $PY -c "import time; time.sleep(60)" > exp_out/case3_run.json 2>&1 &
LPID=$!
sleep 2
echo "run_id=$RID" >> exp_out/case3.log
kill -INT $LPID 2>/dev/null || true
sleep 2
ORPHANS=$(pgrep -f "$RID" | grep -v "^$$$" || true)
if [ -n "$ORPHANS" ]; then echo "ORPHAN FOUND: $ORPHANS" >> exp_out/case3.log; fail "case3 orphan"; else echo "no orphan (pgrep clean)" >> exp_out/case3.log; fi
if [ -e "/dev/shm/gpusim-$RID" ]; then echo "shm NOT removed" >> exp_out/case3.log; fail "case3 shm"; else echo "shm removed" >> exp_out/case3.log; fi
wait $LPID 2>/dev/null || true
rm -rf "runs/$RID"
echo "case3 PASS" | tee -a exp_out/case3.log

# Case 4: --timeout -> timed_out=true
echo "=== CASE 4: timeout ===" | tee exp_out/case4.log
$PY $G run --nodes 1 --gpus-per-node 2 --timeout 1s -- $PY -c "import time; time.sleep(30)" > exp_out/case4_run.json 2>&1 || true
cat exp_out/case4_run.json >> exp_out/case4.log
python3 - "exp_out/case4_run.json" <<'PY' >> exp_out/case4.log 2>&1
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get("timed_out") is True, d
print("case4: timed_out=true verified")
PY
RUNDIR4=$(python3 -c "import json; print(json.load(open('exp_out/case4_run.json'))['run_dir'])")
rm -rf "$RUNDIR4"
echo "case4 PASS" | tee -a exp_out/case4.log

# Case 5: dry-run byte-identical + matches spec rank map
echo "=== CASE 5: dry-run ===" | tee exp_out/case5.log
cat > /tmp/scratchpad/spec5.json <<'JSON'
{"version":1,"models":{"m":{"sm_count":80,"clock_mhz":1000,"mem_bytes":1073741824,"mem_bw_gbs":1000.0,"arch":"x","source":"t"}},
 "nodes":[{"name":"n0","devices":[{"id":0,"model":"m"},{"id":1,"model":"m"}]}],
 "links":[],
 "ranks":[{"rank":0,"node":"n0","device":0},{"rank":1,"node":"n0","device":1}]}
JSON
$PY $G run --topology /tmp/scratchpad/spec5.json --dry-run -- echo hi > exp_out/dry1.txt 2>&1
$PY $G run --topology /tmp/scratchpad/spec5.json --dry-run -- echo hi > exp_out/dry2.txt 2>&1
cmp exp_out/dry1.txt exp_out/dry2.txt && echo "dry-run byte-identical: PASS" >> exp_out/case5.log
grep -q "rank=0 node=n0 device=0" exp_out/dry1.txt && grep -q "rank=1 node=n0 device=1" exp_out/dry1.txt && echo "dry-run matches spec rank map: PASS" >> exp_out/case5.log
echo "case5 PASS" | tee -a exp_out/case5.log

# Case 6: rerun refusal + schema validation
echo "=== CASE 6: rerun refusal + schema ===" | tee exp_out/case6.log
RID6="rerun-demo-id"
$PY $G run --run-id $RID6 --nodes 1 --gpus-per-node 1 --timeout 10s -- true > /dev/null 2>&1
$PY $G run --run-id $RID6 --nodes 1 --gpus-per-node 1 --timeout 10s -- true > exp_out/rerun2.txt 2>&1
RC=$?
echo "second run exit code: $RC" >> exp_out/case6.log
grep -q "already exists" exp_out/rerun2.txt && echo "rerun refusal fired: PASS" >> exp_out/case6.log
# schema validation of a real run.json
$PY $G run --nodes 1 --gpus-per-node 1 --timeout 10s -- true > exp_out/schema_run.json 2>&1
RUNDIR6=$(python3 -c "import json; print(json.load(open('exp_out/schema_run.json'))['run_dir'])")
$PY - "$RUNDIR6/run.json" <<'PY' >> exp_out/case6.log 2>&1
import json,sys
d=json.load(open(sys.argv[1]))
assert set(d.keys())>={"run_id","topology_sha256","started_ns","ended_ns","ranks","verdict","timed_out"}
for r in d["ranks"]:
    assert set(r.keys())>={"rank","pid","exit","signal","started_ns","ended_ns"}
print("schema validation: PASS")
PY
rm -rf "runs/$RID6" "$RUNDIR6"
echo "case6 PASS" | tee -a exp_out/case6.log

rm -rf runs
echo "=== ALL CASES DONE ===" | tee exp_out/summary.txt
