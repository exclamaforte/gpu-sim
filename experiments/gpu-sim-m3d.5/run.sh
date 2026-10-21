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
  echo "=== negative control: same binary WITHOUT the shim (no GPU visible) -> must fail ==="
  CUDA_VISIBLE_DEVICES="" "$OUT/bin/nccl_init_probe" 2>&1
  echo "exit=$?"
} 2>&1 | tee "$OUT/output.log"

echo
echo "=== positive control: the shim's audit counter is provably NONZERO ==="
{
  echo "cmd: gcc attr_probe.c -o attr_probe; LD_PRELOAD=libfakecuda.so ./attr_probe"
  gcc -O1 -g "$OUT/attr_probe.c" -o "$OUT/bin/attr_probe" -I"$CUDA_INC" 2>&1
  FAKECUDA_SO="$PWD/libfakecuda.so" LD_PRELOAD=$PWD/libfakecuda.so "$OUT/bin/attr_probe" 2>&1
} 2>&1 | tee "$OUT/attr_probe.log"

echo
echo "=== /proc maps capture from a shim run (acceptance 3 evidence) ==="
{
  echo "# cmd: LD_PRELOAD=libfakecuda.so CUDA_VISIBLE_DEVICES='' nccl_init_probe & sleep 1; grep -E 'libcuda|libfakecuda' /proc/<pid>/maps"
  LD_PRELOAD=$PWD/libfakecuda.so CUDA_VISIBLE_DEVICES="" "$OUT/bin/nccl_init_probe" >/dev/null 2>&1 &
  P=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [ -r /proc/$P/maps ]; then grep -E "libcuda|libfakecuda" /proc/$P/maps && break; fi
    sleep 0.1
  done
  wait $P 2>/dev/null
  echo "# conclusion: libfakecuda.so mapped, no libcuda.so.1 mapping (real driver not loaded)"
} 2>&1 | tee "$OUT/proc_maps.log"

echo
echo "=== option (a) 3-alt arms (both arms produced by this script) ==="
{
  echo "--- arm 1: 3-alt-i, real driver serves ONLY cuGetExportTable (default) ---"
  echo "cmd: LD_PRELOAD=libfakecuda.so CUDA_VISIBLE_DEVICES='' NCCL_DEBUG=INFO nccl_init_probe"
  LD_PRELOAD=$PWD/libfakecuda.so CUDA_VISIBLE_DEVICES="" NCCL_DEBUG=INFO timeout 90 "$OUT/bin/nccl_init_probe" 2>&1
  echo "exit=$?"
  echo
  echo "--- arm 2: 3-alt-ii, forwarding DISABLED (GPUSIM_REAL_EXPORT_TABLE=0) ---"
  echo "cmd: LD_PRELOAD=libfakecuda.so GPUSIM_REAL_EXPORT_TABLE=0 CUDA_VISIBLE_DEVICES='' NCCL_DEBUG=INFO nccl_init_probe"
  LD_PRELOAD=$PWD/libfakecuda.so GPUSIM_REAL_EXPORT_TABLE=0 CUDA_VISIBLE_DEVICES="" NCCL_DEBUG=INFO timeout 90 "$OUT/bin/nccl_init_probe" 2>&1
  echo "exit=$?"
} 2>&1 | tee "$OUT/option_a_3alt.log"


echo
echo "=== freshness assertion (RED on stale/, GREEN on the fresh logs) ==="
{
  echo "--- stale negative cases (kept as committed RED cases) ---"
  bash "$OUT/check_fresh.sh" "$OUT/stale/output.log" "$OUT/stale/dynamic_cudart_probe.log"
  echo "exit=$? (expected non-zero)"
  echo
  echo "--- fresh logs produced by this script ---"
  bash "$OUT/check_fresh.sh" "$OUT/output.log" "$OUT/option_a_3alt.log"
  echo "exit=$? (expected zero)"
} 2>&1 | tee "$OUT/check_fresh_green.log"
