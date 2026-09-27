#!/usr/bin/env bash
# gpu-sim-u2o.6.1.1 -- diagnose the real-GPU cudaMalloc failure
# ("OS call failed or operation not supported on this OS") seen in job
# 20260927t052025-1f7067db. Isolates: harness rlimits vs. host memory pressure.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
TMP=$(mktemp -d)
cd "$REPO"

cat > "$TMP/probe.cu" <<'EOF'
#include <stdio.h>
#include <cuda_runtime.h>
int main(void) {
    cudaError_t e = cudaFree(0);
    printf("{\"stage\":\"ctx\",\"rc\":%d,\"msg\":\"%s\"}\n", (int)e, cudaGetErrorString(e));
    fflush(stdout);
    void* p = NULL;
    e = cudaMalloc(&p, 1024);
    printf("{\"stage\":\"malloc1k\",\"rc\":%d,\"msg\":\"%s\",\"ptr\":%p}\n", (int)e, cudaGetErrorString(e), p);
    fflush(stdout);
    int dev=0; cudaGetDevice(&dev);
    cudaDeviceProp pr; e = cudaGetDeviceProperties(&pr, dev);
    printf("{\"stage\":\"props\",\"rc\":%d,\"name\":\"%s\",\"totalMemMB\":%zu,\"canMapHost\":%d}\n",
           (int)e, e==cudaSuccess?pr.name:"?", (size_t)(pr.totalGlobalMem>>20), (int)pr.canMapHostMemory);
    fflush(stdout);
    return e == cudaSuccess ? 0 : 1;
}
EOF

nvcc -O2 -arch=sm_120 "$TMP/probe.cu" -o "$TMP/probe" || exit 9

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "--- host memory ---"; free -m
echo "--- A: no rlimits ---"; "$TMP/probe"; echo "A rc=$?"

echo "--- B: harness ResourceLimits (AS=256G, DATA=8G, CPU=60s) ---"
python3 - "$TMP/probe" <<'EOF'
import resource, subprocess, sys
def apply():
    for r, v in ((resource.RLIMIT_AS, 262144*1024*1024), (resource.RLIMIT_CPU, 60),
                 (resource.RLIMIT_DATA, 8192*1024*1024), (resource.RLIMIT_CORE, 0)):
        resource.setrlimit(r, (v, v))
sys.exit(subprocess.run([sys.argv[1]], preexec_fn=apply).returncode)
EOF
echo "B rc=$?"

echo "--- C: kernel_driver-synthesized scale program, no rlimits ---"
python3 -c "
import sys; sys.path.insert(0, '$REPO')
from epic_u2o.kernel_driver import spec_from_stage1, synthesize
open('$TMP/scale.cu','w').write(synthesize(spec_from_stage1('scale',[4,1,1],[256,1,1],1024)))
" || exit 8
nvcc -O2 -arch=sm_120 "$TMP/scale.cu" -o "$TMP/scale" || exit 9
"$TMP/scale" | head -3; echo "C rc=${PIPESTATUS[0]}"

echo "--- D: same binary via SafeHarness.run_kernel (harness rlimits) ---"
SCALE_EXE="$TMP/scale" python3 -c "
import os, sys; sys.path.insert(0, '$REPO')
from epic_u2o.safe_harness import SafeHarness
r = SafeHarness(timeout_sec=60).run_kernel(os.environ['SCALE_EXE'])
print('D rc=%s timeout=%s stderr=%r' % (r.returncode, r.timeout, r.stderr[:200]))
print('D stdout_head=%r' % (r.stdout.splitlines()[:2],))
"
echo "=== diag done ==="
rm -rf "$TMP"
