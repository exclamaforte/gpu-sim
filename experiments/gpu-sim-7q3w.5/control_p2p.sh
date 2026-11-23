#!/usr/bin/env bash
# gpu-sim-7q3w.5 p2p arm: acceptance run + isolation + NEGATIVE control.
#
# The p2p arm failed before this bead's change with rank1 wedged in the fabric
# poll loop (run-20261122T155004: "poll_cq -> 0 (rings out=49 in=48)" forever).
# The diagnosis was that inbound records were only copied into a process's
# memory inside poll_cq, so NCCL's p2p send setup -- which spins on a host-memory
# FIFO that the peer's RDMA WRITE must fill -- deadlocked against its own
# receiver.  The fix is an asynchronous pump thread (models the NIC writing
# memory on its own; FAKEIB_NO_PUMP_THREAD=1 disables it).
#
# This script runs three phases so the claim is a measurement, not a theory:
#   A acceptance : allreduce + p2p, pump ON        -> must PASS
#   B isolation  : p2p alone,            pump ON   -> must PASS
#   C control    : p2p alone,            pump OFF  -> must WEDGE (no p2p line)
#
# Run through the GPU queue:
#   ts --max-runtime 10m -L '7q3w.5 p2p acceptance + negative control' -- \
#       bash experiments/gpu-sim-7q3w.5/control_p2p.sh
set -uo pipefail

ROOT=${GPU_SIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
EXP="$ROOT/experiments/gpu-sim-7q3w.5"
OUT="$ROOT/.lab/tmp/build-7q3w5-numerics/control-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$OUT"

note() { echo "--- $*"; }
failures=0
chk() { if eval "$1"; then echo "ASSERT ok: $2"; else echo "ASSERT FAIL: $2"; failures=$((failures+1)); fi; }

note "control dir $OUT"

# ---------------------------------------------------------------- phase A ----
note "A: allreduce + p2p, async pump ON (acceptance shape)"
ARM_P2P=1 bash "$EXP/run.sh" >"$OUT/A-acceptance.log" 2>&1
rcA=$?
echo "$rcA" >"$OUT/A.rc"
chk "[[ $rcA -eq 0 ]]" "A: run.sh exited 0"
chk "grep -q 'NUMERICS allreduce rank=0 .* bad=0$' '$OUT/A-acceptance.log'" \
    "A: allreduce exact on rank0"
chk "grep -q 'NUMERICS allreduce rank=1 .* bad=0$' '$OUT/A-acceptance.log'" \
    "A: allreduce exact on rank1"
chk "grep -q 'NUMERICS p2p rank=1 nbytes=4096 mismatches=0$' '$OUT/A-acceptance.log'" \
    "A: 4096-byte send/recv byte-identical"
chk "grep -q 'NUMERICS RUN OK: exact 2-rank allreduce AND byte-identical' '$OUT/A-acceptance.log'" \
    "A: run reported both claims"

# ---------------------------------------------------------------- phase B ----
note "B: p2p alone (no allreduce first), async pump ON"
ARM_P2P=1 ARM_SKIP_AR=1 bash "$EXP/run.sh" >"$OUT/B-p2p-only.log" 2>&1
rcB=$?
echo "$rcB" >"$OUT/B.rc"
chk "[[ $rcB -eq 0 ]]" "B: p2p does not depend on an earlier allreduce"
chk "grep -q 'NUMERICS p2p rank=1 nbytes=4096 mismatches=0$' '$OUT/B-p2p-only.log'" \
    "B: 4096-byte send/recv byte-identical without a prior collective"

# ---------------------------------------------------------------- phase C ----
# The control must FAIL for the documented reason: the receive side never gets
# the payload because the sender's host-memory FIFO spin is never filled.  A
# 45 s rank timeout is enough -- the wedge is a permanent deadlock, not a slow
# path (the same run completes in ~2 s with the pump).
note "C: p2p alone, pump DISABLED (negative control; expected to wedge)"
ARM_P2P=1 ARM_SKIP_AR=1 FAKEIB_NO_PUMP_THREAD=1 ARM_TRACE=1 RANK_TIMEOUT=45 \
    bash "$EXP/run.sh" >"$OUT/C-no-pump.log" 2>&1
rcC=$?
echo "$rcC" >"$OUT/C.rc"
# On failure run.sh prints "run dir kept at <dir>"; on success "RUN_DIR=<dir>".
CRUN=$(sed -n 's/^.*run dir kept at //p; s/^RUN_DIR=//p' "$OUT/C-no-pump.log" | tail -1)
echo "$CRUN" >"$OUT/C.run_dir"
chk "[[ $rcC -ne 0 ]]" "C: without the pump the run fails (exit $rcC)"
chk "[[ -n '$CRUN' && -s '$CRUN/rank1.rc' ]] && grep -qx 124 '$CRUN/rank1.rc'" \
    "C: rank1 hit the rank timeout instead of receiving the payload"
chk "[[ -n '$CRUN' ]] && ! grep -q 'NUMERICS p2p rank=1' '$CRUN/rank1.log'" \
    "C: rank1 never printed a received payload"
chk "[[ -n '$CRUN' ]] && grep -q 'poll_cq .* -> 0 (rings out=' '$CRUN/rank1.log'" \
    "C: rank1 was wedged in the fabric poll loop (the 7q3w.3 signature)"
chk "grep -q 'ASSERT FAIL: rank1: 0 of 4096 received bytes' '$OUT/C-no-pump.log'" \
    "C: the missing payload is reported as the p2p failure, not hidden"

echo
if (( failures )); then
    echo "P2P CONTROL FAILED: $failures assertion(s); control dir kept at $OUT"
    exit 1
fi
echo "P2P CONTROL OK: acceptance and isolation pass WITH the async inbound pump; the same p2p arm wedges WITHOUT it"
echo "CONTROL_DIR=$OUT"
