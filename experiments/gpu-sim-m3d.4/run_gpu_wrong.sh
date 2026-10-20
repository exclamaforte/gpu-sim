#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$PATH"
OUT="$HERE/nccl_debug_wrong.log"
: > "$HERE/trace_wrong.log"
export NCCL_NET_PLUGIN="$HERE/build/libncclNetSim_wrong.so"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export GPUSIM_TRACE="$HERE/trace_wrong.log"
export GPUSIM_NET_DEVICES=1
echo "=== wrong-symbol plugin run $(date -Is) ===" > "$OUT"
stdbuf -o0 -e0 "$HERE/build/probe_load" >> "$OUT" 2>&1
rc=$?
echo "wrong exit=$rc" >> "$OUT"
echo "done wrong=$rc"