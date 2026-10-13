/* gpu-sim-urr launch-cost bench (acceptance 3 + 4), experiments-only.
 *
 * Timing evidence does not belong in make check: it measures wall-clock
 * sleep + lookup under load. Margins are wide on purpose (see run.sh).
 * Two modes, both device-free (direct cudaLaunchKernel calls, libfakegpu):
 *
 *   scalar N     no resolver; FAKEGPU_KERNEL_LATENCY_MS from env. Asserts
 *                nothing itself; run.sh checks total >= 0.9 * N * LAT_MS
 *                (nanosleep never returns early -- a LOWER bound only;
 *                overshoot under load is fine and expected).
 *   resolver0 N  zero-resolver installed; the SAME launches must take a
 *                small fraction of the scalar expectation (the resolver
 *                path REPLACED the scalar sleep), and the per-launch mean
 *                is the lookup-cost number for acceptance 4.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda_runtime.h"
#include "fakegpu.h"

static char key_dev[8], key_host[8];

static long zero_resolver(const char *kernel_name,
                          int gridX, int gridY, int gridZ,
                          int blockX, int blockY, int blockZ,
                          unsigned int sharedMem)
{
    (void)kernel_name; (void)gridX; (void)gridY; (void)gridZ;
    (void)blockX; (void)blockY; (void)blockZ; (void)sharedMem;
    return 0;
}

static long now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000000L + (long)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s scalar|resolver0 N\n", argv[0]);
        return 2;
    }
    int resolver_mode = strcmp(argv[1], "resolver0") == 0;
    int n = atoi(argv[2]);
    if (n < 1) return 2;

    __cudaRegisterFunction(NULL, key_host, key_dev, "bench_kernel", 0,
                           NULL, NULL, NULL, NULL, NULL);
    if (resolver_mode)
        fg_cfg()->cost_resolver = zero_resolver;

    long t0 = now_ns();
    for (int i = 0; i < n; i++) {
        cudaError_t e = cudaLaunchKernel(key_dev,
                                         8, 1, 1, 256, 1, 1, 0,
                                         NULL, NULL, NULL);
        if (e != cudaSuccess) {
            printf("FAIL launch %d: %d\n", i, (int)e);
            return 1;
        }
    }
    long dt = now_ns() - t0;
    printf("mode=%s n=%d total_ns=%ld per_launch_ns=%.1f\n", argv[1], n, dt,
           (double)dt / (double)n);
    return 0;
}
