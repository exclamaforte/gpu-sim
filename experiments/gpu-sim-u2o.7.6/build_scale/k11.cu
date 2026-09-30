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

__global__ void k_random_access(const float* a, float* y, const unsigned int* seed, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned s=seed[i]*1664525u+1013904223u;y[i]=a[s%n];}}

extern "C" int u2o_run_11(void) {
    float *d_a = NULL;
    float *h_a = (float*)malloc(N * sizeof(float));
    float *d_y = NULL;
    float *h_y = (float*)malloc(N * sizeof(float));
    unsigned int *d_seed = NULL;
    unsigned int *h_seed = (unsigned int*)malloc(N * sizeof(unsigned int));
    CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));
    CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_seed, N * sizeof(unsigned int)));
    CHECK(cudaMemcpy(d_seed, h_seed, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
    for (int i = 0; i < N; i++) h_a[i] = (float)(i * 0.5f + 1.0f);
    for (int i = 0; i < N; i++) h_y[i] = (float)0;
    for (int i = 0; i < N; i++) h_seed[i] = (unsigned int)(i * 0.5f + 1.0f);
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    for (int w = 0; w < 5; w++) { k_random_access<<<grid, block, 0>>>(d_a, d_y, d_seed, N); }
    CHECK(cudaDeviceSynchronize());
    for (int t = 0; t < 20; t++) {
        float ms = 0.0f;
        CHECK(cudaEventRecord(start));
        k_random_access<<<grid, block, 0>>>(d_a, d_y, d_seed, N);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("{\"trial\": %d, \"time_ms\": %.6f}\n", t, ms);
        fflush(stdout);
    }
    CHECK(cudaDeviceSynchronize());
    cudaFree(d_a); free(h_a);
    cudaFree(d_y); free(h_y);
    cudaFree(d_seed); free(h_seed);
    return 0;
}
