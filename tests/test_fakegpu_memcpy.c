/*
 * test_fakegpu_memcpy.c — gpu-sim-2qe regression test.
 *
 * THE BUG THIS PINS (src/fakegpu_cuda.c, before the fix):
 *     cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, int kind) {
 *         (void)src; fg_simulate_memcpy(n); fg_write_random(dst, n); return cudaSuccess;
 *     }
 * and the same for cudaMemcpyAsync / HtoD / DtoH / DtoD / HtoDAsync, plus
 * cudaMemset / cudaMemsetD8, which did (void)value; fg_fill(...).
 *
 * So every copy and every fill silently produced xorshift noise while returning
 * cudaSuccess. Value correctness was structurally impossible, and a round-trip
 * comparison is the only thing that can see it.
 *
 * Each check below compares BYTES back against what was written. Under the old
 * code every one of them fails (the destination is noise, matched only by
 * astronomically unlikely chance); under the fix they pass exactly.
 *
 * Exit 0 = pass. Non-zero = the primitive still loses data.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 4096

static int failures = 0;

static void fail(const char *what, size_t i, unsigned char got, unsigned char want)
{
    fprintf(stderr, "FAIL %s at byte %zu: got 0x%02x want 0x%02x\n", what, i, got, want);
    failures++;
}

/* Deterministic non-uniform pattern: no two adjacent bytes equal, so a shifted
 * copy is caught as well as a random one. */
static void write_pattern(unsigned char *p, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (unsigned char)((i * 31u + seed * 17u + 5u) & 0xFF);
}

static void check_bytes(const char *what, const unsigned char *got,
                        const unsigned char *want, size_t n, int *already)
{
    for (size_t i = 0; i < n; i++) {
        if (got[i] != want[i] && *already < 3) {
            fail(what, i, got[i], want[i]);
            (*already)++;
        }
    }
    if (*already) failures += 1;
}

int main(void)
{
    unsigned char *devA = NULL, *devB = NULL;
    unsigned char *hostA = NULL, *hostB = NULL;
    unsigned char expect[N];

    if (cudaMalloc((void **)&devA, N) != cudaSuccess) { fprintf(stderr, "cudaMalloc A failed\n"); return 2; }
    if (cudaMalloc((void **)&devB, N) != cudaSuccess) { fprintf(stderr, "cudaMalloc B failed\n"); return 2; }
    hostA = (unsigned char *)malloc(N);
    hostB = (unsigned char *)malloc(N);
    if (!hostA || !hostB) { fprintf(stderr, "host malloc failed\n"); return 2; }

    /* ---- 1. cudaMemcpy DtoD round trip -------------------------------- */
    write_pattern(devA, N, 1);
    memcpy(expect, devA, N);                 /* devA is host memory in this emulator */
    memset(devB, 0, N);
    if (cudaMemcpy(devB, devA, N, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DtoD returned an error\n"); return 2;
    }
    int seen = 0;
    check_bytes("cudaMemcpy(DtoD)", devB, expect, N, &seen);

    /* ---- 2. cudaMemcpyDtoD named entry point -------------------------- */
    memset(devB, 0, N);
    if (cudaMemcpyDtoD(devB, devA, N) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDtoD returned an error\n"); return 2;
    }
    seen = 0;
    check_bytes("cudaMemcpyDtoD", devB, expect, N, &seen);

    /* ---- 3. cudaMemcpyHtoD then cudaMemcpyDtoH ------------------------ */
    write_pattern(hostA, N, 2);
    memset(devB, 0, N);
    if (cudaMemcpyHtoD(devB, hostA, N) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHtoD returned an error\n"); return 2;
    }
    seen = 0;
    check_bytes("cudaMemcpyHtoD", devB, hostA, N, &seen);

    memset(hostB, 0, N);
    if (cudaMemcpyDtoH(hostB, devB, N) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDtoH returned an error\n"); return 2;
    }
    seen = 0;
    check_bytes("cudaMemcpyDtoH", hostB, hostA, N, &seen);

    /* ---- 4. cudaMemcpyAsync ------------------------------------------- */
    memset(devB, 0, N);
    if (cudaMemcpyAsync(devB, devA, N, cudaMemcpyDeviceToDevice, 0) != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync returned an error\n"); return 2;
    }
    seen = 0;
    check_bytes("cudaMemcpyAsync", devB, expect, N, &seen);

    /* ---- 5. cudaMemset must write the requested BYTE ------------------ */
    for (int v = 0; v <= 0xA5; v += 0xA5) {
        memset(devB, 0x5A, N);               /* pre-dirty so a no-op is visible */
        if (cudaMemset(devB, v, N) != cudaSuccess) {
            fprintf(stderr, "cudaMemset returned an error\n"); return 2;
        }
        int bad = 0;
        for (size_t i = 0; i < N; i++) {
            if (devB[i] != (unsigned char)v) {
                if (bad < 3) fail("cudaMemset", i, devB[i], (unsigned char)v);
                bad++;
            }
        }
        if (bad) failures++;
    }

    /* ---- 6. cudaMemsetD8 --------------------------------------------- */
    memset(devB, 0x00, N);
    if (cudaMemsetD8(devB, 0x3C, N) != cudaSuccess) {
        fprintf(stderr, "cudaMemsetD8 returned an error\n"); return 2;
    }
    {
        int bad = 0;
        for (size_t i = 0; i < N; i++) {
            if (devB[i] != 0x3C) {
                if (bad < 3) fail("cudaMemsetD8", i, devB[i], 0x3C);
                bad++;
            }
        }
        if (bad) failures++;
    }

    free(hostA); free(hostB);
    cudaFree(devA); cudaFree(devB);

    if (failures) {
        fprintf(stderr, "test_fakegpu_memcpy: FAILED (%d check group(s))\n", failures);
        return 1;
    }
    printf("test_fakegpu_memcpy: PASS (copy + fill round trips are byte-exact)\n");
    return 0;
}
