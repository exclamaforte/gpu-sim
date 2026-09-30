extern "C" int N;
extern "C" dim3 grid;
extern "C" dim3 block;
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "{\"error\": \"%s:%d %s\"} \n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 2; } } while (0)

__global__ void k_atomic_hist(const float* a, unsigned* h, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int b=(int)fabsf(a[i]*16.0f);if(b<256)atomicAdd(&h[b],1u);}}

extern "C" int u2o_run_24(void) {
    float *d_a = NULL;
    float *h_a = (float*)malloc(N * sizeof(float));
    unsigned int *d_h = NULL;
    unsigned int *h_h = (unsigned int*)malloc(N * sizeof(unsigned int));
    CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_h, N * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_h, h_h, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
    for (int i = 0; i < N; i++) h_a[i] = (float)(i * 0.5f + 1.0f);
    for (int i = 0; i < N; i++) h_h[i] = (unsigned int)(i * 0.5f + 1.0f);
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    for (int w = 0; w < 5; w++) { k_atomic_hist<<<grid, block, 0>>>(d_a, d_h, N); }
    CHECK(cudaDeviceSynchronize());
    for (int t = 0; t < 20; t++) {
        float ms = 0.0f;
        CHECK(cudaEventRecord(start));
        k_atomic_hist<<<grid, block, 0>>>(d_a, d_h, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("{\"trial\": %d, \"time_ms\": %.6f}\n", t, ms);
        fflush(stdout);
    }
    CHECK(cudaDeviceSynchronize());
    cudaFree(d_a); free(h_a);
    cudaFree(d_h); free(h_h);
    return 0;
}
