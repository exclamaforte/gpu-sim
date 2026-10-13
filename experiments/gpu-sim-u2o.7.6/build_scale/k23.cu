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

__global__ void k_warp_reduce(const float* a, float* y, int n){for(int b=blockIdx.x*blockDim.x; b<n; b+=blockDim.x*gridDim.x){float v=(b+threadIdx.x<n)?a[b+threadIdx.x]:0.0f;for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o);if(threadIdx.x==0&&b<n)y[b]=v;}}

extern "C" int u2o_run_23(void) {
    float *d_a = NULL;
    float *h_a = (float*)malloc(N * sizeof(float));
    float *d_y = NULL;
    float *h_y = (float*)malloc(N * sizeof(float));
    CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));
    CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < N; i++) h_a[i] = (float)(i * 0.5f + 1.0f);
    for (int i = 0; i < N; i++) h_y[i] = (float)0;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    for (int w = 0; w < 5; w++) { k_warp_reduce<<<grid, block, 0>>>(d_a, d_y, N); }
    CHECK(cudaDeviceSynchronize());
    for (int t = 0; t < 20; t++) {
        float ms = 0.0f;
        CHECK(cudaEventRecord(start));
        k_warp_reduce<<<grid, block, 0>>>(d_a, d_y, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("{\"trial\": %d, \"time_ms\": %.6f}\n", t, ms);
        fflush(stdout);
    }
    CHECK(cudaDeviceSynchronize());
    cudaFree(d_a); free(h_a);
    cudaFree(d_y); free(h_y);
    return 0;
}
