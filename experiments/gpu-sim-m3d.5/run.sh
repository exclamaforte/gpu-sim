#!/usr/bin/env bash
# gpu-sim-m3d.5: build the fake driver shim and run the driver-seam probe.
# CPU-only: no GPU is used and no ts job is required.
set -uo pipefail
cd "$(dirname "$0")/../.."
NCCL_INC=/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include
NCCL_LIB=/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib
CUDA_INC=/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include
DYNCUDART_LIB=${M3D5_DYNCUDART_LIB:-/tmp/ncclsrc/nccl-2.32.3-1/build/lib}
OUT=experiments/gpu-sim-m3d.5

make libfakecuda.so >/dev/null || exit 1
mkdir -p "$OUT/bin"
gcc -O1 -g experiments/gpu-sim-m3d.5/nccl_init_probe.c -o "$OUT/bin/nccl_init_probe" \
    -I"$NCCL_INC" -I"$CUDA_INC" -L"$NCCL_LIB" -l:libnccl.so.2 -Wl,-rpath,"$NCCL_LIB" || exit 1

{
  echo "=== probe A: wheel NCCL (static cudart) WITH shim ==="
  LD_PRELOAD=$PWD/libfakecuda.so CUDA_VISIBLE_DEVICES="" "$OUT/bin/nccl_init_probe" 2>&1
  echo "exit=$?"
  echo
  echo "=== probe B: from-source NCCL (dynamic cudart) WITH shim ==="
  echo "    lib: $DYNCUDART_LIB/libnccl.so.2.32.3"
  LD_PRELOAD=$PWD/libfakecuda.so CUDA_VISIBLE_DEVICES="" \
    LD_LIBRARY_PATH="$DYNCUDART_LIB:/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/lib" \
    "$OUT/bin/nccl_init_probe" 2>&1
  echo "exit=$?"
  echo
  echo "=== negative control: same binary WITHOUT the shim (no GPU visible) ==="
  CUDA_VISIBLE_DEVICES="" "$OUT/bin/nccl_init_probe" 2>&1
  echo "exit=$?"
  echo
  echo "=== negative control: WITHOUT the shim, real GPU visible (1 device) ==="
  "$OUT/bin/nccl_init_probe" 2>&1
  echo "exit=$?"
} 2>&1 | tee "$OUT/output.log"
