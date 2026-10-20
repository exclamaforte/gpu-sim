#!/usr/bin/env bash
set -uo pipefail
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
wait $P0; R0=$?
wait $P1; R1=$?
echo "rank0 exit=$R0 rank1 exit=$R1"
echo "=== rank0.log ==="; cat "$HERE/rank0.log"
echo "=== rank1.log ==="; cat "$HERE/rank1.log"
rm -f "$UIDFILE"