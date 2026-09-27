/*
 * reference_kernel.cu -- FIXED reference kernel for the noise-floor
 * measurement (gpu-sim-u2o.6.6.1).
 *
 * Deliberately NOT generated and NOT untrusted: this is hand-written,
 * reviewed-by-eye code with no shared memory, no atomics, no dynamic shapes.
 * Nothing here depends on the harness's fake path, the generator, or any
 * stage in rework.
 *
 * Two kernels, so launch cost can be separated from kernel cost:
 *
 *   k_empty  -- does nothing. A launch of this measures the *launch path*
 *               (kernel dispatch + whatever the timing harness adds).
 *   k_scale  -- the reference workload: out[i] = in[i] * alpha, memory
 *               bound, the same family the stage-1 dataset uses.
 *
 * Two timing modes, because the difference between them IS the answer:
 *
 *   single  -- cudaEventRecord / one launch / cudaEventRecord. This is what
 *              kernel_driver.synthesize() does, and therefore what every
 *              stage-1 measurement to date has measured.
 *   batched -- cudaEventRecord / K launches / cudaEventRecord, divided by K.
 *              Amortises the per-measurement event and launch overhead.
 *
 * If `single` and `batched` medians agree, the per-launch cost is real
 * hardware dispatch. If `single` is much larger, the reported number was
 * mostly measurement overhead, and the fix is measurement design rather
 * than better statistics.
 *
 * Usage: reference_kernel <empty|scale> <N> <block> <trials> <batch K>
 * Prints one JSON line per measurement:
 *   {"mode":"single","trial":0,"us":2.610,"median_us":...}
 * and a final summary line:
 *   {"summary":true,"mode":"single","n":200,"median_us":...,"p05_us":...,
 *    "p95_us":...,"min_us":...,"max_us":...}
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                        \
    do {                                                                   \
        cudaError_t e_ = (call);                                            \
        if (e_ != cudaSuccess) {                                           \
            fprintf(stderr, "{\"error\": \"%s:%d %s\"}\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(e_));           \
            return 2;                                                      \
        }                                                                  \
    } while (0)

__global__ void k_empty(void) { }

__global__ void k_scale(const float *in, float *out, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * alpha;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc < 6) {
        fprintf(stderr, "usage: %s <empty|scale> <N> <block> <trials> <batchK>\n", argv[0]);
        return 1;
    }
    const char *mode = argv[1];
    int is_empty = (strcmp(mode, "empty") == 0);
    int n = atoi(argv[2]);
    int block = atoi(argv[3]);
    int trials = atoi(argv[4]);
    int batch = atoi(argv[5]);
    if (n < 1 || block < 1 || trials < 1 || batch < 1) {
        fprintf(stderr, "{\"error\":\"bad args\"}\n");
        return 1;
    }
    int grid = (n + block - 1) / block;
    if (grid < 1) grid = 1;
    if (grid > 65535) grid = 65535;   /* classic launch bound, x-dim only */

    float *d_in = NULL, *d_out = NULL;
    float *h_in = NULL;
    if (!is_empty) {
        CHECK(cudaMalloc((void **)&d_in, (size_t)n * sizeof(float)));
        CHECK(cudaMalloc((void **)&d_out, (size_t)n * sizeof(float)));
        h_in = (float *)malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) h_in[i] = (float)(i & 1023);
        CHECK(cudaMemcpy(d_in, h_in, (size_t)n * sizeof(float),
                         cudaMemcpyHostToDevice));
    }

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

#define LAUNCH()                                                          \
    do {                                                                  \
        if (is_empty) k_empty<<<grid, block>>>();                         \
        else         k_scale<<<grid, block>>>(d_in, d_out, 2.0f, n);      \
    } while (0)

    for (int w = 0; w < 20; w++) LAUNCH();
    CHECK(cudaDeviceSynchronize());

    double *us = (double *)malloc((size_t)trials * sizeof(double));
    for (int t = 0; t < trials; t++) {
        float ms = 0.0f;
        CHECK(cudaEventRecord(start, 0));
        for (int b = 0; b < batch; b++) LAUNCH();
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        us[t] = (double)ms * 1000.0 / (double)batch;   /* us per launch */
        printf("{\"mode\":\"%s\",\"trial\":%d,\"batch\":%d,\"us\":%.6f}\n",
               is_empty ? "empty" : "scale", t, batch, us[t]);
        fflush(stdout);
    }

    /* summary: median and percentiles of the per-launch times */
    double *sorted = (double *)malloc((size_t)trials * sizeof(double));
    memcpy(sorted, us, (size_t)trials * sizeof(double));
    qsort(sorted, (size_t)trials, sizeof(double), cmp_d);
    int p05 = (int)(trials * 0.05), p95 = (int)(trials * 0.95);
    if (p95 >= trials) p95 = trials - 1;
    if (p05 < 0) p05 = 0;
    printf("{\"summary\":true,\"mode\":\"%s\",\"kernel\":\"%s\",\"n\":%d,"
           "\"block\":%d,\"grid\":%d,\"batch\":%d,\"trials\":%d,"
           "\"median_us\":%.6f,\"p05_us\":%.6f,\"p95_us\":%.6f,"
           "\"min_us\":%.6f,\"max_us\":%.6f}\n",
           is_empty ? "empty" : "scale", is_empty ? "k_empty" : "k_scale",
           n, block, grid, batch, trials,
           sorted[trials / 2], sorted[p05], sorted[p95],
           sorted[0], sorted[trials - 1]);
    fflush(stdout);

    if (!is_empty) {
        CHECK(cudaFree(d_in));
        CHECK(cudaFree(d_out));
        free(h_in);
    }
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    free(us);
    free(sorted);
    return 0;
}
