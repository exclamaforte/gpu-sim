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

__global__ void k_tiled_gemm(const float* A,const float* B,float* C,int D){__shared__ float As[32][32],Bs[32][32];int t=blockIdx.x*32+threadIdx.x;for(int k=0;k<32;k+=1){As[threadIdx.x][k]=A[t*32+k];Bs[threadIdx.x][k]=B[k*32+threadIdx.x];}__syncthreads();for(int k=0;k<32;k++)C[t*32+threadIdx.x]+=As[threadIdx.x][k]*Bs[threadIdx.x][k];}

extern "C" int u2o_run_12(void) {
    float *d_A = NULL;
    float *h_A = (float*)malloc(N * sizeof(float));
    float *d_B = NULL;
    float *h_B = (float*)malloc(N * sizeof(float));
    float *d_C = NULL;
    float *h_C = (float*)malloc(N * sizeof(float));
    CHECK(cudaMalloc((void**)&d_A, N * sizeof(float)));
    CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_B, N * sizeof(float)));
    CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&d_C, N * sizeof(float)));
    CHECK(cudaMemcpy(d_C, h_C, N * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < N; i++) h_A[i] = (float)(i * 0.5f + 1.0f);
    for (int i = 0; i < N; i++) h_B[i] = (float)(i * 0.5f + 1.0f);
    for (int i = 0; i < N; i++) h_C[i] = (float)(i * 0.5f + 1.0f);
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    for (int w = 0; w < 5; w++) { k_tiled_gemm<<<grid, block, 0>>>(d_A, d_B, d_C, 1); }
    CHECK(cudaDeviceSynchronize());
    for (int t = 0; t < 20; t++) {
        float ms = 0.0f;
        CHECK(cudaEventRecord(start));
        k_tiled_gemm<<<grid, block, 0>>>(d_A, d_B, d_C, 1);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("{\"trial\": %d, \"time_ms\": %.6f}\n", t, ms);
        fflush(stdout);
    }
    CHECK(cudaDeviceSynchronize());
    cudaFree(d_A); free(h_A);
    cudaFree(d_B); free(h_B);
    cudaFree(d_C); free(h_C);
    return 0;
}
