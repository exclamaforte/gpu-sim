#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
cd "$(git rev-parse --show-toplevel)"
ROOT="$(pwd)"
TMPROOT="${FU0H_TMP_ROOT:-$ROOT}"
EXP="$ROOT/experiments/gpu-sim-7q3w.1"
LIBDIR="$EXP/libs/cu12"
WHEELDIR="$EXP/wheels"
NCCLINC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
CUDAINC="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
CUDALIB="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib"
# fu0h-reaper-start
# Reap build dirs left by SIGKILLED runs (only when the owning PID is dead).
for _d in "$TMPROOT"/.lab/tmp/7q3w-cu12-build-*; do
  [ -d "$_d" ] || continue
  _pid="${_d##*7q3w-cu12-build-}"
  case "$_pid" in ''|*[!0-9]*) continue ;; esac
  if ! kill -0 "$_pid" 2>/dev/null; then
    rm -rf "$_d"
  fi
done
# fu0h-reaper-end
BUILDDIR="$TMPROOT/.lab/tmp/7q3w-cu12-build-$$"
mkdir -p "$BUILDDIR"
PLUGIN="$BUILDDIR/libncclNetPlugin.so"
PROBE="$BUILDDIR/probe_load_cu12"
trap 'rm -rf "$BUILDDIR"' EXIT
# project-local wheel cache
(
  mkdir -p "$LIBDIR"
  exec 9>"$LIBDIR/.extract.lock"
  flock -w 300 9 || { echo "fu0h: timed out waiting for extraction lock" >&2; exit 1; }
  if [ ! -s "$LIBDIR/libnccl.so.2" ] || [ "$(stat -c%s "$LIBDIR/libnccl.so.2" 2>/dev/null || echo 0)" -lt 100000000 ]; then
  mkdir -p "$LIBDIR/nvidia/nccl/lib" "$WHEELDIR"
  if ! ls "$WHEELDIR"/nvidia_nccl_cu12-*.whl >/dev/null 2>&1; then
    pip download nvidia-nccl-cu12 --no-deps -d "$WHEELDIR" >/dev/null 2>&1
  fi
  WHL=$(ls "$WHEELDIR"/nvidia_nccl_cu12-*.whl | head -1)
  cd "$LIBDIR"
  unzip -o "$WHL" "nvidia/nccl/lib/libnccl.so*" >/dev/null 2>&1
  cp nvidia/nccl/lib/libnccl.so.2 libnccl.so.2
  cd "$ROOT"
  fi
)
ln -sf libnccl.so.2 "$LIBDIR/libnccl.so"
# build plugin (same recovered layout)
gcc -shared -fPIC -O0 -Wno-incompatible-pointer-types -o "$PLUGIN" "$EXP/plugin_v11.c"
# build probe against cu12
gcc -I"$NCCLINC" -I"$CUDAINC" -L"$LIBDIR" -L"$CUDALIB" \
    -o "$PROBE" \
    "$ROOT/experiments/gpu-sim-465.1/probe_load.c" -lnccl -lcudart
NCCL_NET_PLUGIN="$PLUGIN" LD_LIBRARY_PATH="$LIBDIR:$CUDALIB:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
  "$PROBE" 2>&1 | sed "s|$ROOT/||g"
