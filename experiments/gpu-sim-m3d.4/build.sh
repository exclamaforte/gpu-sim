#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_INC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
CUDA_INC="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
NCCL_LIB="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/lib"
VENDOR="$HERE/vendor"
mkdir -p "$HERE/build"
CXXINC="/usr/include/c++/16"
g++ -shared -fPIC -O2 -g -std=c++20 -Wall -I"$VENDOR" -I"$NCCL_INC" -I"$CUDA_INC" -I"$CXXINC" \
    "$HERE/src/ncclnetsim.c" -o "$HERE/build/libncclNetSim.so"
g++ -shared -fPIC -O2 -g -std=c++20 -Wall -DWRONG_SYMBOL=1 -I"$VENDOR" -I"$NCCL_INC" -I"$CUDA_INC" -I"$CXXINC" \
    "$HERE/src/ncclnetsim.c" -o "$HERE/build/libncclNetSim_wrong.so"
echo "built:"
ls -la "$HERE/build"
echo "exported symbols (correct):"
nm -D "$HERE/build/libncclNetSim.so" | grep -i "ncclNetPlugin\|ncclNet_v"