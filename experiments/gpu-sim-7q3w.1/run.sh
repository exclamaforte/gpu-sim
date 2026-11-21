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
EXP="$ROOT/experiments/gpu-sim-7q3w.1"
LIBDIR="$EXP/libs/cu13"
NCCLINC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
CUDAINC="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
CUDALIB="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib"
WHEELDIR="$EXP/wheels"
BUILDDIR="$ROOT/.lab/tmp/7q3w-build-$$"
mkdir -p "$BUILDDIR"
PLUGIN="$BUILDDIR/libncclNetPlugin.so"
PROBE="$BUILDDIR/probe_load"
trap 'rm -rf "$BUILDDIR"' EXIT
# project-local wheel cache
(
  mkdir -p "$LIBDIR"
  exec 9>"$LIBDIR/.extract.lock"
  flock 9
  if [ ! -s "$LIBDIR/libnccl.so.2" ] || [ "$(stat -c%s "$LIBDIR/libnccl.so.2" 2>/dev/null || echo 0)" -lt 100000000 ]; then
  mkdir -p "$LIBDIR/nvidia/nccl/lib" "$WHEELDIR"
  if ! ls "$WHEELDIR"/nvidia_nccl_cu13-*.whl >/dev/null 2>&1; then
    pip download nvidia-nccl-cu13 --no-deps -d "$WHEELDIR" >/dev/null 2>&1
  fi
  WHL=$(ls "$WHEELDIR"/nvidia_nccl_cu13-*.whl | head -1)
  cd "$LIBDIR"
  unzip -o "$WHL" "nvidia/nccl/lib/libnccl.so*" >/dev/null 2>&1
  cp nvidia/nccl/lib/libnccl.so.2 libnccl.so.2
  ln -sf libnccl.so.2 libnccl.so
  cd "$ROOT"
  fi
)
# build plugin + probe
gcc -shared -fPIC -O0 -Wno-incompatible-pointer-types -o "$PLUGIN" "$EXP/plugin_v11.c"
gcc -I"$NCCLINC" -I"$CUDAINC" -L"$LIBDIR" -L"$CUDALIB" \
    -o "$PROBE" \
    "$ROOT/experiments/gpu-sim-465.1/probe_load.c" -lnccl -lcudart

echo "=== POSITIVE (NCCL_DEBUG=INFO) ==="
set +e
NCCL_NET_PLUGIN="$PLUGIN" LD_LIBRARY_PATH="$LIBDIR:$CUDALIB:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV \
  "$PROBE" 2>&1 | sed "s|$ROOT/||g"
INFO_RC=${PIPESTATUS[0]}
set -e
echo "INFO_EXIT=$INFO_RC"

echo "=== NEGATIVE (NCCL_DEBUG=WARN) ==="
set +e
NCCL_NET_PLUGIN="$PLUGIN" LD_LIBRARY_PATH="$LIBDIR:$CUDALIB:${LD_LIBRARY_PATH:-}" \
  NCCL_DEBUG=WARN \
  "$PROBE" 2>&1 | sed "s|$ROOT/||g"
WARN_RC=${PIPESTATUS[0]}
set -e
echo "WARN_EXIT=$WARN_RC"
