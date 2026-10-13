/* gpu-sim-m64(b): the SHIM sizeof guard, compiled against shim headers.
 *
 * TWO cudaDeviceProp LAYOUTS: shim/cuda_runtime.h is 696 bytes, real CUDA
 * 13.3 is 1008. This file is built by gcc with -Ishim (Makefile tests/%),
 * so it sees the SHIM struct -- the layout the fake's OWN field writes use
 * (prop->maxThreadsPerBlock = 1024 style in src/fakegpu_cuda.c). A field
 * added to the shim struct moves those writes; this assertion fails loudly
 * instead of letting 16 unpinned fields slide silently (the u2o.16 SIGFPE
 * failure mode against real PyTorch).
 *
 * The other side -- the real-layout overlay's 19 absolute-offset writes --
 * is guarded by tests/test_device_prop_overlay.cu (real nvcc, real headers,
 * sizeof == 1008). Each assertion fails only for its own side.
 *
 * If this sizeof changes, that is a FINDING about the shim layout: report
 * it, do not silently accommodate it by editing the constant.
 */
#include <stdio.h>
#include "cuda_runtime.h"

int main(void)
{
    size_t got = sizeof(struct cudaDeviceProp);
    if (got != 696) {
        printf("FAIL sizeof(struct cudaDeviceProp) = %lu, want 696 -- "
               "the shim layout moved; the fake's field writes are suspect\n",
               (unsigned long)got);
        return 1;
    }
    printf("ok   sizeof(struct cudaDeviceProp) = %lu (shim layout)\n",
           (unsigned long)got);
    return 0;
}
