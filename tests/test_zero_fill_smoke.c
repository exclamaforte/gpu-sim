/*
 * test_zero_fill_smoke.c — plumbing smoke test for the fake-GPU interposer.
 *
 * Exercises the full H2D -> launch -> D2H pipeline. The fake GPU never executes
 * kernels, so the D2H buffer is zero-filled (MAP_ANONYMOUS pages). This test
 * verifies the interposer does not leak random bytes into device memory.
 * It is NOT compute coverage: no kernel runs, no arithmetic is checked.
 *
 * Run:  LD_PRELOAD=./libfakegpu.so ./build/vector_add
 */
#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Fake kernel symbol — in real CUDA this would be __global__; here we pass a
 * dummy function pointer (the interposer never dereferences it). */
static const void *vector_add_kernel = (const void *)0x1;

#define N 1024

static int run(void)
{
    float *h_a = (float *)malloc(sizeof(float) * N);
    float *h_b = (float *)malloc(sizeof(float) * N);
    float *h_c = (float *)malloc(sizeof(float) * N);
    if (!h_a || !h_b || !h_c) { fprintf(stderr, "host alloc failed\n"); return 1; }

    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
        h_c[i] = 0.0f;
    }

    void *d_a, *d_b, *d_c;
    if (cudaMalloc(&d_a, sizeof(float) * N) != cudaSuccess)  return 1;
    if (cudaMalloc(&d_b, sizeof(float) * N) != cudaSuccess)  return 1;
    if (cudaMalloc(&d_c, sizeof(float) * N) != cudaSuccess)  return 1;

    if (cudaMemcpyHtoD(d_a, h_a, sizeof(float) * N) != cudaSuccess) return 1;
    if (cudaMemcpyHtoD(d_b, h_b, sizeof(float) * N) != cudaSuccess) return 1;

    int nbytes = (int)(sizeof(float) * N);
    void *params[4] = { &d_a, &d_b, &d_c, &nbytes };
    if (cudaLaunchKernel(vector_add_kernel, 1, 1, 1, N, 1, 1, 0, NULL,
                         params, NULL) != cudaSuccess)
        return 1;

    if (cudaMemcpyDtoH(h_c, d_c, sizeof(float) * N) != cudaSuccess) return 1;

    /* Value-level check (gpu-sim-2qe): the fake GPU does not execute the
       kernel, so h_c stays zero-filled from cudaMalloc (mmap zero pages).
       Verify the D2H round-trip preserved zeros rather than random bytes. */
    int bad = 0;
    for (int i = 0; i < N; i++)
        if (h_c[i] != 0.0f) { bad = 1; break; }
    if (bad) { fprintf(stderr, "FAIL: D2H buffer not zero-filled (random data leak)\n"); return 1; }
    printf("test_zero_fill_smoke: N=%d  D2H buffer zero-filled (no random leak)\n", N);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}

int main(void)
{
    int dev = -1;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev < 1) {
        fprintf(stderr, "vector_add: no devices\n");
        return 1;
    }
    printf("vector_add: fake GPUs visible = %d\n", dev);
    return run();
}
