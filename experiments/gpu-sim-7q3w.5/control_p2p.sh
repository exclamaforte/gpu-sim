#!/usr/bin/env bash
# gpu-sim-7q3w.5 p2p control: acceptance, isolation, MECHANISM control, repeats.
#
# The p2p arm failed before this bead's change with rank1 wedged in the fabric
# poll loop (run-20261122T155004: "poll_cq -> 0 (rings out=49 in=48)" forever).
# Diagnosis, measured: inbound records were only copied into a process's memory
# inside poll_cq, so NCCL's p2p send setup -- which spins on a host-memory FIFO
# the peer's RDMA WRITE must fill -- deadlocked against its own receiver.
#
# PHASES (each repeated NREP times; every run records rc, wall, run dir, and
# the observables below):
#   A acceptance : allreduce + p2p, pump started at device open -> must PASS
#   B isolation  : p2p alone,       pump started at device open -> must PASS
#   C no pump    : p2p alone,       no progress thread          -> must WEDGE
#   D lazy pump  : p2p alone,       FAKEIB_LAZY_PUMP=1 (pre-fix: thread starts
#                                   on the first poll_cq)       -> must WEDGE
# C shows the pump is NECESSARY; D shows STARTING IT AT DEVICE OPEN is the fix
# (same binary, same arm, only the start point differs).  B carries trace so the
# passing case shows the sender POSTING the payload, which C/D cannot.
#
# Observables recorded per run (all from the shim's own trace):
#   rank1_posts_conn    rank1's 64-byte connection info reached the wire
#   rank0_sees_inbound  rank0 delivered anything from the ring at all
#   rank0_posts_payload rank0 posted the 4096-byte data WR
#   rank1_gets_payload  rank1 delivered the 4096-byte payload
#   rank1_p2p_line      rank1 printed a received payload (the RESULT)
#
# Run through the GPU queue:
#   ts --max-runtime 10m -L '7q3w.5 p2p control N=5' -- \
#       bash experiments/gpu-sim-7q3w.5/control_p2p.sh
set -uo pipefail

ROOT=${GPU_SIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
EXP="$ROOT/experiments/gpu-sim-7q3w.5"
OUT="$ROOT/.lab/tmp/build-7q3w5-numerics/control-$(date +%Y%m%dT%H%M%S)"
NREP=${NREP:-5}
WEDGE_TIMEOUT=${WEDGE_TIMEOUT:-20}   # rank timeout for C/D; B finishes in ~3.5 s
mkdir -p "$OUT"
: >"$OUT/records.jsonl"

note() { echo "--- $*"; }

# one <phase> <rep> <env...>: run run.sh, record the facts, keep the log
one() {
    local phase=$1 rep=$2; shift 2
    local log="$OUT/$phase-$rep.log"
    local t0 t1 rc wall rundir rc0 rc1
    local rank1_posts_conn=0 rank0_sees_inbound=0 rank0_posts_payload=0
    local rank1_gets_payload=0 rank1_p2p_line=0
    t0=$(date +%s.%N)
    env "$@" bash "$EXP/run.sh" >"$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
    rundir=$(sed -n 's/^RUN_DIR=//p; s/^.*run dir kept at //p' "$log" | tail -1)
    rc0=""; rc1=""
    if [[ -n "$rundir" ]]; then
        [[ -s "$rundir/rank0.rc" ]] && rc0=$(cat "$rundir/rank0.rc")
        [[ -s "$rundir/rank1.rc" ]] && rc1=$(cat "$rundir/rank1.rc")
        if [[ -f "$rundir/rank0.log" ]]; then
            grep -q 'deliver_inbound' "$rundir/rank0.log" && rank0_sees_inbound=1
            grep -q 'post_send .*len=4096' "$rundir/rank0.log" && rank0_posts_payload=1
        fi
        if [[ -f "$rundir/rank1.log" ]]; then
            grep -q 'ring_put(1,0) kind=2 nbytes=64' "$rundir/rank1.log" && rank1_posts_conn=1
            grep -q 'deliver_inbound .*nbytes=4096' "$rundir/rank1.log" && rank1_gets_payload=1
            grep -q 'NUMERICS p2p rank=1' "$rundir/rank1.log" && rank1_p2p_line=1
        fi
    fi
    printf '{"phase":"%s","rep":%d,"rc":%d,"wall_s":%s,"run_dir":"%s","rank0_rc":"%s","rank1_rc":"%s","rank1_posts_conn":%d,"rank0_sees_inbound":%d,"rank0_posts_payload":%d,"rank1_gets_payload":%d,"rank1_p2p_line":%d}\n' \
        "$phase" "$rep" "$rc" "$wall" "$rundir" "$rc0" "$rc1" \
        "$rank1_posts_conn" "$rank0_sees_inbound" "$rank0_posts_payload" \
        "$rank1_gets_payload" "$rank1_p2p_line" >>"$OUT/records.jsonl"
    printf '   %s rep %d: rc=%d wall=%ss rank0_rc=%s rank1_rc=%s conn=%d saw=%d post=%d got=%d\n' \
        "$phase" "$rep" "$rc" "$wall" "${rc0:-?}" "${rc1:-?}" \
        "$rank1_posts_conn" "$rank0_sees_inbound" "$rank0_posts_payload" \
        "$rank1_gets_payload"
}

note "control dir $OUT (nrep=$NREP, wedge timeout=${WEDGE_TIMEOUT}s)"
for i in $(seq 1 "$NREP"); do
    note "A$i: allreduce + p2p, pump at device open (acceptance)"
    one A "$i" ARM_P2P=1
    note "B$i: p2p alone, pump at device open (isolation; trace on)"
    one B "$i" ARM_P2P=1 ARM_SKIP_AR=1 ARM_TRACE=1
    note "C$i: p2p alone, NO progress thread (pump necessary)"
    one C "$i" ARM_P2P=1 ARM_SKIP_AR=1 FAKEIB_NO_PUMP_THREAD=1 \
        ARM_TRACE=1 RANK_TIMEOUT="$WEDGE_TIMEOUT"
    note "D$i: p2p alone, LAZY pump start = pre-fix shape (mechanism)"
    one D "$i" ARM_P2P=1 ARM_SKIP_AR=1 FAKEIB_LAZY_PUMP=1 \
        ARM_TRACE=1 RANK_TIMEOUT="$WEDGE_TIMEOUT"
done

printf '{"config":{"nrep":%d,"wedge_timeout_s":%d,"run_sh":"experiments/gpu-sim-7q3w.5/run.sh"},"records":[' \
    "$NREP" "$WEDGE_TIMEOUT" >"$OUT/manifest.json"
paste -sd, "$OUT/records.jsonl" >>"$OUT/manifest.json"
printf ']}\n' >>"$OUT/manifest.json"

echo
echo "P2P CONTROL DONE: $((NREP * 4)) runs; records in $OUT/records.jsonl"
echo "CONTROL_DIR=$OUT"
