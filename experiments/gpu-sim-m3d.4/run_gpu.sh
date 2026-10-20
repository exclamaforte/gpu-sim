#!/usr/bin/env bash
# gpu-sim-m3d.4 GPU job body (ts only). Runs correct + wrong-symbol plugin.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:$PATH"

# --- correct plugin ---
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

# --- wrong-symbol control ---
OUT2="$HERE/nccl_debug_wrong.log"
: > "$HERE/trace_wrong.log"
export NCCL_NET_PLUGIN="$HERE/build/libncclNetSim_wrong.so"
export GPUSIM_TRACE="$HERE/trace_wrong.log"
echo "=== wrong-symbol plugin run $(date -Is) ===" > "$OUT2"
stdbuf -o0 -e0 "$HERE/build/probe_load" >> "$OUT2" 2>&1
rc2=$?
echo "wrong exit=$rc2" >> "$OUT2"

echo "done correct=$rc wrong=$rc2"