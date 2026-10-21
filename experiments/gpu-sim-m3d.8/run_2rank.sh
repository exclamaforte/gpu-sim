#!/usr/bin/env bash
set -uo pipefail
# Reap the ranks if this harness is killed. A killed harness must NOT leave
# orphaned ranks spinning: on 2026-08-22 two probe ranks survived two hours
# at 77% CPU each on this lab's 2-CPU quota, which contaminated another bead's
# timing measurements and had to be killed by the PI. SIGINT, SIGTERM and a
# normal exit all run this.
RANKS=()
cleanup() {
  for p in "${RANKS[@]:-}"; do
    [ -n "$p" ] && kill -TERM "$p" 2>/dev/null
  done
  sleep 0.2
  for p in "${RANKS[@]:-}"; do
    [ -n "$p" ] && kill -KILL "$p" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$PATH"
RUNID="m3d8-$$"
UIDFILE="/tmp/m3d8-uid-$$"
rm -f "$UIDFILE"
export GPUSIM_RUN_ID="$RUNID"
export GPUSIM_NRANKS=2
export GPUSIM_TIMEOUT_NS=5000000
export NCCL_NET_PLUGIN="$HERE/build/libncclNetSimFab.so"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export GPUSIM_TRACE="$HERE/trace.log"
: > "$HERE/trace.log"
export NCCL_MAX_NCHANNELS=1
export NCCL_MULTI_RANK_GPU_ENABLE=1

# rank0
GPUSIM_RANK=0 "$HERE/build/allreduce_test" 0 2 "$UIDFILE" > "$HERE/rank0.log" 2>&1 &
P0=$!
# rank1
GPUSIM_RANK=1 "$HERE/build/allreduce_test" 1 2 "$UIDFILE" > "$HERE/rank1.log" 2>&1 &
P1=$!
RANKS+=("$P0" "$P1")
wait $P0; R0=$?
wait $P1; R1=$?
echo "rank0 exit=$R0 rank1 exit=$R1"
echo "=== rank0.log ==="; cat "$HERE/rank0.log"
echo "=== rank1.log ==="; cat "$HERE/rank1.log"
rm -f "$UIDFILE"