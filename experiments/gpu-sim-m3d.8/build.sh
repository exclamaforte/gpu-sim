#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NCCL_INC="/home/gabe/.local/lib/python3.14/site-packages/nvidia/nccl/include"
CUDA_INC="/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include"
VENDOR="$HERE/vendor"
FABRIC_SRC="$REPO/src/fakegpu_fabric.c"
FABRIC_INC="$REPO/include"
mkdir -p "$HERE/build"
# compile fabric as C
gcc -c -O2 -g -fPIC -I"$FABRIC_INC" "$FABRIC_SRC" -o "$HERE/build/fabric.o"
# compile plugin as C++
g++ -shared -fPIC -O2 -g -std=c++20 -Wall -I"$VENDOR" -I"$NCCL_INC" -I"$CUDA_INC" -I"$FABRIC_INC" \
    "$HERE/src/ncclnetsim_fabric.c" "$HERE/build/fabric.o" -o "$HERE/build/libncclNetSimFab.so" -lpthread
echo "built libncclNetSimFab.so"
nm -D "$HERE/build/libncclNetSimFab.so" | grep -i "ncclNetPlugin"