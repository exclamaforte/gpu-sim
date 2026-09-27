/*
 * test_cudart_none.cu — proves the "-cudart=none" fake path is complete.
 *
 * gpu-sim-u2o.6.1.3. This file is compiled by nvcc with -cudart=none and
 * linked against libfakegpu ONLY:
 *
 *   nvcc -cudart=none -arch=sm_120 tests/test_cudart_none.cu \
 *        -o <exe> -L. -lfakegpu -Xlinker -rpath -Xlinker <repo>
 *
 * The resulting binary links neither libcudart nor libcuda, so it cannot
 * touch a real GPU even in principle, and every cudart symbol it needs --
 * including the __cuda* registration/launch symbols nvcc's generated host
 * stub references -- must resolve inside libfakegpu. There is no
 * per-symbol fallthrough to a real CUDA library, which is exactly the
 * partial-fake mixing that u2o.16 diagnosed and this epic forbids.
 *
 * It exercises a __global__ kernel, a <<<>>> launch, cudaMalloc,
 * cudaMemcpy, cudaEventRecord/ElapsedTime/Synchronize, cudaGetLastError
 * and cudaDeviceSynchronize, and prints one JSON line per trial so the
 * runner can assert the fake's configured latency shows up.
 *
 * NOT tested here, by design: device memory CONTENTS. **global** bodies
 * never execute in an API-surface fake, so a real CUDA program's output
 * values are not reproduced -- only the API traffic, launch validation and
 * timing. Anything a caller expects to read back from device memory stays
 * fake-by-design.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                        \
    do {                                                                  \
        cudaError_t e_ = (call);                                           \
        if (e_ != cudaSuccess) {                                          \
            fprintf(stderr, "{\"error\": \"%s:%d %s\"}\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(e_));          \
            return 2;                                                     \
        }                                                                 \
    } while (0)

__global__ void scale_kernel(const float *in, float *out, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * alpha;
}

int main(void)
{
    const int N = 1024;
    dim3 grid(4, 1, 1);
    dim3 block(256, 1, 1);
    float *d_in = NULL, *d_out = NULL;
    float *h_in = (float *)malloc(N * sizeof(float));
    float *h_out = (float *)malloc(N * sizeof(float));
    cudaEvent_t start, stop;

    CHECK(cudaMalloc((void **)&d_in, N * sizeof(float)));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i * 0.5f + 1.0f);
    CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void **)&d_out, N * sizeof(float)));

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    /* error-state API: a fake has none, so this must be success */
    CHECK(cudaGetLastError());
    CHECK(cudaPeekAtLastError());

    for (int w = 0; w < 5; w++) scale_kernel<<<grid, block>>>(d_in, d_out, 2.0f, N);
    CHECK(cudaDeviceSynchronize());

    for (int t = 0; t < 3; t++) {
        float ms = 0.0f;
        CHECK(cudaEventRecord(start, 0));
        scale_kernel<<<grid, block>>>(d_in, d_out, 2.0f, N);
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("{\"trial\": %d, \"time_ms\": %.6f}\n", t, ms);
        fflush(stdout);
    }

    /* an invalid launch must be rejected, not silently accepted */
    {
        dim3 bad(0, 1, 1);
        cudaError_t v = cudaGetLastError();
        (void)v;
        scale_kernel<<<bad, block>>>(d_in, d_out, 2.0f, N);
        cudaError_t e = cudaDeviceSynchronize();
        printf("{\"bad_launch_rejected\": %s}\n",
               (e != cudaSuccess) ? "true" : "false");
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    return 0;
}
