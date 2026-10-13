/* gpu-sim-urr functional attachment test (acceptance 1, 2, 5, 6, 7).
 *
 * No nvcc, no device, no timing: it calls the fake's own entry points
 * directly with chosen pointers, so every assertion is deterministic.
 * Built by gcc against the SHIM headers (Makefile tests/%) and linked
 * against libfakegpu.
 *
 * 1. Registers two known names under chosen deviceFun/hostFun keys, launches
 *    with each key, asserts the simulator's resolver saw THAT name.
 * 2. Spy resolver records the full (name, grid, block, shmem) tuple.
 * 5. Same grid/block launched under two names -> the resolver receives two
 *    different names: the declared inputs carry what disambiguation needs
 *    (the artifact, keyed by name, is 9.1's job, not this bead's).
 * 6. cost_resolver lives in struct fg_config (offsetof in-bounds) and NOT in
 *    cudaDeviceProp (shim sizeof re-witnessed 696 here; m64 guards both).
 * 7. D_GPUSIM: the resolver is consumed as a pure function of launch params
 *    -- repeated identical launches yield identical spy inputs, and the
 *    registry holds no per-launch accumulation. Structurally it cannot write
 *    back: it returns long ns and takes no out-params, no record access
 *    (see fg_simulate_kernel_resolved: the return flows only to sleep).
 */
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include "cuda_runtime.h"
#include "fakegpu.h"

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (cond) { printf("  ok   " __VA_ARGS__); printf("\n"); } \
    else { printf("  FAIL "); printf(__VA_ARGS__); printf("\n"); failures++; } \
} while (0)

/* Distinct addresses standing in for deviceFun/hostFun keys. */
static char key_dev_alpha[8], key_host_alpha[8];
static char key_dev_beta[8], key_host_beta[8];
static char key_unknown[8];

#define MAX_CALLS 32
static struct {
    char name[64];
    int gx, gy, gz, bx, by, bz;
    unsigned int shmem;
} calls[MAX_CALLS];
static int n_calls = 0;

/* Spy resolver: per-name ns table keyed EXACTLY like a model artifact would
 * be (name + launch params select; grid*block never implies problem_size). */
static long spy_resolver(const char *kernel_name,
                         int gridX, int gridY, int gridZ,
                         int blockX, int blockY, int blockZ,
                         unsigned int sharedMem)
{
    if (n_calls < MAX_CALLS) {
        snprintf(calls[n_calls].name, sizeof(calls[0].name), "%s",
                 kernel_name ? kernel_name : "(null)");
        calls[n_calls].gx = gridX; calls[n_calls].gy = gridY;
        calls[n_calls].gz = gridZ; calls[n_calls].bx = blockX;
        calls[n_calls].by = blockY; calls[n_calls].bz = blockZ;
        calls[n_calls].shmem = sharedMem;
        n_calls++;
    }
    if (kernel_name && strcmp(kernel_name, "kernel_alpha") == 0) return 5000;
    if (kernel_name && strcmp(kernel_name, "kernel_beta") == 0) return 9000;
    return 1000;
}

static void register_both(const char *host, const char *dev, const char *name)
{
    __cudaRegisterFunction(NULL, host, (char *)dev, name, 0,
                           NULL, NULL, NULL, NULL, NULL);
}

int main(void)
{
    struct fg_config *c = fg_cfg();
    c->cost_resolver = spy_resolver;

    register_both(key_host_alpha, key_dev_alpha, "kernel_alpha");
    register_both(key_host_beta, key_dev_beta, "kernel_beta");

    /* 1+2: launch by deviceFun key; the simulator must see name + full tuple. */
    cudaError_t e = cudaLaunchKernel(key_dev_alpha,
                                     8, 1, 1, 256, 1, 1, 0, NULL, NULL, NULL);
    CHECK(e == cudaSuccess, "launch by deviceFun succeeds");
    CHECK(n_calls == 1, "resolver consumed exactly one launch");
    CHECK(strcmp(calls[0].name, "kernel_alpha") == 0,
          "deviceFun key resolves to kernel_alpha");
    CHECK(calls[0].gx == 8 && calls[0].bx == 256 && calls[0].shmem == 0,
          "grid/block/shmem pass through (8,256,0)");

    /* HostFun key (the stub path __cudaLaunchKernel/__cudaGetKernel use). */
    e = cudaLaunchKernel(key_host_alpha,
                         8, 1, 1, 256, 1, 1, 0, NULL, NULL, NULL);
    CHECK(e == cudaSuccess, "launch by hostFun succeeds");
    CHECK(n_calls == 2 && strcmp(calls[1].name, "kernel_alpha") == 0,
          "hostFun key resolves to kernel_alpha");

    /* 5: identical grid/block, different name -> different resolver input. */
    e = cudaLaunchKernel(key_dev_beta,
                         8, 1, 1, 256, 1, 1, 0, NULL, NULL, NULL);
    CHECK(e == cudaSuccess, "second kernel launches");
    CHECK(n_calls == 3 && strcmp(calls[2].name, "kernel_beta") == 0,
          "same (8,256,0), different name reaches resolver");

    /* Unregistered func: scalar default, resolver NOT consulted. */
    int before = n_calls;
    e = cudaLaunchKernel(key_unknown,
                         8, 1, 1, 256, 1, 1, 0, NULL, NULL, NULL);
    CHECK(e == cudaSuccess, "unregistered func still launches");
    CHECK(n_calls == before, "unregistered func takes scalar default");

    /* urr rework: the MISS contract, asserted directly. find_name must
     * return NULL -- never a garbage pointer, never "" (empty would make
     * every unresolvable launch look like a kernel called "", and the
     * resolver's default path distinguishes NULL from a real name). The
     * launch-level check above could not catch this: reading UB happened
     * to behave. */
    CHECK(fg_handle_find_name(key_unknown) == NULL,
          "miss returns NULL, not garbage");
    CHECK(fg_handle_find_name(NULL) == NULL,
          "NULL key returns NULL");
    CHECK(fg_handle_find_name(key_dev_alpha) != NULL &&
          strcmp(fg_handle_find_name(key_dev_alpha), "kernel_alpha") == 0,
          "hit still resolves after the miss-path fix");

    /* 7: D_GPUSIM -- the resolver is a pure function of launch params from
     * the sim's side: repeat the identical launch, get identical inputs,
     * with nothing accumulated anywhere in between. */
    e = cudaLaunchKernel(key_dev_alpha,
                         8, 1, 1, 256, 1, 1, 0, NULL, NULL, NULL);
    CHECK(e == cudaSuccess, "repeat launch succeeds");
    CHECK(n_calls == before + 1, "repeat launch consumed once");
    CHECK(strcmp(calls[n_calls - 1].name, "kernel_alpha") == 0 &&
          calls[n_calls - 1].gx == 8 && calls[n_calls - 1].bx == 256,
          "repeat launch delivers identical inputs (no feedback)");

    /* 6: the resolver lives in fg_config, never in cudaDeviceProp. */
    CHECK(offsetof(struct fg_config, cost_resolver) < sizeof(struct fg_config),
          "cost_resolver inside fg_config");
    CHECK(sizeof(struct cudaDeviceProp) == 696,
          "shim cudaDeviceProp still 696 (resolver NOT there)");
    printf("  info fg_config sizeof = %lu\n", (unsigned long)sizeof(struct fg_config));

    if (failures) {
        printf("\n%d check(s) FAILED\n", failures);
        return 1;
    }
    printf("\nresolver attach gate PASSED\n");
    return 0;
}
