#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/gpu-sim-6kp
NCCL=/tmp/ncclsrc/nccl-2.32.3-1/build
CUDA=/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux
OUT=/tmp/scratchpad/6kp_run
mkdir -p "$OUT"

# build probes
cc -O1 -g -Iinclude -Ishim -o "$OUT/probe_0a" $EXP/probe_0a.c -L. -lfakegpu -Wl,-rpath,$(pwd)
NCCL=$NCCL CUDA=$CUDA cc -O1 -g -I$NCCL/include -I$CUDA/include -Iinclude -Ishim \
  -o "$OUT/probe_nccl4" $EXP/probe_nccl4.c -L. -lfakegpu -L$NCCL/lib -lnccl \
  -Wl,-rpath,$(pwd) -Wl,-rpath,$NCCL/lib -Wl,-rpath-link,$CUDA/lib

export GPUSIM_TOPOLOGY=/tmp/topo_4dev.bin
export LD_LIBRARY_PATH=$NCCL/lib:$CUDA/lib

# probe 0a
LD_PRELOAD=$(pwd)/libfakegpu.so "$OUT/probe_0a" > "$OUT/probe_0a.out" 2>&1
grep -q "libfakegpu.so" "$OUT/probe_0a.out"
grep -q "cudaGetDeviceCount() = 4" "$OUT/probe_0a.out"
echo "probe_0a: PASS"

# 4-rank init (bounded)
LD_PRELOAD=$(pwd)/libfakegpu.so timeout 120 "$OUT/probe_nccl4" 4 "$OUT/maps4" > "$OUT/probe_nccl4.out" 2>&1
grep -q "ALL RANKS OK" "$OUT/probe_nccl4.out"
echo "probe_nccl4: PASS"

# diff against committed artifacts
diff <(sed -E "s/0x[0-9a-f]+/<ADDR>/g" "$OUT/probe_0a.out") <(sed -E "s/0x[0-9a-f]+/<ADDR>/g" "$EXP/probe_0a_output.txt") >/dev/null && echo "probe_0a output: matches committed (addresses normalized)" || echo "probe_0a output: DRIFT"
diff <(cat "$OUT/probe_nccl4.out") "$EXP/probe_nccl4_output.txt" >/dev/null && echo "probe_nccl4 output: matches committed" || echo "probe_nccl4 output: DRIFT"

echo "6kp run.sh: OK"