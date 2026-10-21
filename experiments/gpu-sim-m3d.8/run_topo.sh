#!/usr/bin/env bash
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
export GPUSIM_RUN_ID="topo-$$"
export GPUSIM_NRANKS=2
export GPUSIM_TIMEOUT_NS=5000000
export NCCL_NET_PLUGIN="$HERE/build/libncclNetSimFab.so"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_TOPO_FILE="$HERE/topo2.xml"
export NCCL_MULTI_RANK_GPU_ENABLE=1
export NCCL_MAX_NCHANNELS=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
UIDFILE="/tmp/m3d8-uid-topo"; rm -f "$UIDFILE"
GPUSIM_RANK=0 "$HERE/build/allreduce_test" 0 2 "$UIDFILE" > "$HERE/topo0.log" 2>&1 &
P0=$!
GPUSIM_RANK=1 "$HERE/build/allreduce_test" 1 2 "$UIDFILE" > "$HERE/topo1.log" 2>&1 &
P1=$!
RANKS+=("$P0" "$P1")
wait $P0; R0=$?
wait $P1; R1=$?
echo "exit0=$R0 exit1=$R1"
grep -i "via" "$HERE/topo0.log"