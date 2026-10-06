#!/usr/bin/env bash
# gpu-sim-8g6 probe job -- JOB BODY, ts ONLY. GPU-idle-fitting, one shot.
# Compiles the per-member probe plugin + the real-allreduce loader probe,
# runs the NCCL loader path with NCCL_DEBUG=INFO, commits log verbatim.
# SIGSEGV (exit 139) is diagnostic, not failure -- do NOT gate on exit code.
# usage, inside ts: bash probe_job.sh <absolute-outdir>
set -uo pipefail
OUTDIR="${1:?usage: probe_job.sh <absolute-outdir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC465="$HERE/../gpu-sim-465.1"
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
NCCL_INC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUTDIR"
echo "=== probe start $(date -Is) ==="
gcc -shared -fPIC -g -O0 "$HERE/probe_v11_markers.c" -o "$OUTDIR/libncclNetProbe.so" \
  || { echo "PLUGIN COMPILE FAILED"; exit 10; }
nvcc "$SRC465/probe_load.c" -o "$OUTDIR/probe_load" -I"$NCCL_INC" -L"$NCCL_LIB" -l:libnccl.so.2 \
  || { echo "LOADER COMPILE FAILED"; exit 11; }
nm -n "$OUTDIR/libncclNetProbe.so" | grep -E " stub_" > "$OUTDIR/stub_symbols.txt"
export NCCL_NET_PLUGIN="$OUTDIR/libncclNetProbe.so"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export U2O_MARKER="$OUTDIR/markers.jsonl"
export LD_LIBRARY_PATH="$NCCL_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
: > "$OUTDIR/markers.jsonl"
"$OUTDIR/probe_load" > "$OUTDIR/nccl_debug.log" 2>&1
rc=$?
echo "probe exit=$rc (139=SIGSEGV, diagnostic)"
echo "--- markers ---"
cat "$OUTDIR/markers.jsonl"
echo "--- stub symbol map ---"
cat "$OUTDIR/stub_symbols.txt"
echo "=== probe done $(date -Is) exit=$rc ==="
exit 0
