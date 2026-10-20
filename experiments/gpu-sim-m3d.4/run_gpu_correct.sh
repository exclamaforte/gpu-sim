#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$PATH"
OUT="$HERE/nccl_debug.log"
: > "$HERE/trace_correct.log"
export NCCL_NET_PLUGIN="$HERE/build/libncclNetSim.so"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export GPUSIM_TRACE="$HERE/trace_correct.log"
export GPUSIM_NET_DEVICES=1
echo "=== correct plugin run $(date -Is) ===" > "$OUT"
stdbuf -o0 -e0 "$HERE/build/probe_load" >> "$OUT" 2>&1
rc=$?
echo "correct exit=$rc" >> "$OUT"
echo "done correct=$rc"