/* gpu-sim-urr identity probe: is register's deviceFun the launch's func?
 *
 * Built by nvcc -cudart=none against the REAL headers and linked against
 * libfakegpu only (same recipe as test_device_prop_overlay.cu). No device is
 * touched: the fake answers registration and launch. Run with
 * FAKEGPU_DEBUG_REGISTRY=1 and compare the [fg-register] pointers against
 * the [fg-launch] pointer in stderr. Exit status only reports launch rc;
 * the verdict lives in the trace (see experiments/gpu-sim-urr/run.sh).
 */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void urr_probe_kernel(float *x)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (x && i == 0) x[0] = 1.0f;
}

__global__ void urr_second_kernel(float *x)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (x && i == 0) x[0] = 2.0f;
}

int main(void)
{
    /* Two different kernels, two different configs: the trace must show two
     * distinct registered names resolving at their own launches. sharedMem
     * nonzero on the second launch exercises the shmem plumbing. */
    urr_probe_kernel<<<8, 256>>>(nullptr);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("FAIL first launch: %s\n", cudaGetErrorString(e));
        return 1;
    }
    urr_second_kernel<<<3, 128, 1024>>>(nullptr);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("FAIL second launch: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("identity probe launches returned success\n");
    return 0;
}
