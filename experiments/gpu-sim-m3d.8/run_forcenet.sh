#!/usr/bin/env bash
if [ -z "${M3D8_SCRIPT_TIMEOUT:-}" ]; then export M3D8_SCRIPT_TIMEOUT=1; exec timeout 60s "$0" "$@"; fi
set -uo pipefail
# Reap the ranks if this harness is killed (see run_2rank.sh for why: a killed
# harness must not leave orphaned ranks spinning on a shared CPU quota).
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
export GPUSIM_RUN_ID="forcenet-$$"
export GPUSIM_NRANKS=2
export GPUSIM_TIMEOUT_NS=5000000
export GPUSIM_ATTACH_TIMEOUT_NS=5000000000
export GPUSIM_DTYPE=float32
export NCCL_NET_PLUGIN="$HERE/build/libncclNetSimFab.so"
export NCCL_DEBUG=INFO
export GPUSIM_TRACE="${GPUSIM_TRACE:-$HERE/trace_default.log}"
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4
UIDFILE="/tmp/m3d8-uid-fn"; rm -f "$UIDFILE" "$GPUSIM_TRACE"
timeout 30s env NCCL_HOSTID="node0" GPUSIM_RANK=0 "$HERE/build/allreduce_test" 0 2 "$UIDFILE" > "$HERE/forcenet0.log" 2>&1 &
P0=$!
timeout 30s env NCCL_HOSTID="node1" GPUSIM_RANK=1 "$HERE/build/allreduce_test" 1 2 "$UIDFILE" > "$HERE/forcenet1.log" 2>&1 &
P1=$!
RANKS+=("$P0" "$P1")
wait $P0; R0=$?
wait $P1; R1=$?
echo "exit0=$R0 exit1=$R1"
grep -i "via\|Using network\|network sim" "$HERE/forcenet0.log"
