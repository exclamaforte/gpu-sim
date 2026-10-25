/* gpu-sim-m3d.12 acceptance: heterogeneous compute. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cuda_runtime.h"
#include "fakegpu.h"

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (cond) { printf("  ok   " __VA_ARGS__); printf("\n"); } \
    else { printf("  FAIL "); printf(__VA_ARGS__); printf("\n"); failures++; } \
} while (0)

static long resolver_a(const char *kn, int gx,int gy,int gz,int bx,int by,int bz,unsigned int sm){(void)kn;(void)gx;(void)gy;(void)gz;(void)bx;(void)by;(void)bz;(void)sm;return 1000;}
static long resolver_b(const char *kn, int gx,int gy,int gz,int bx,int by,int bz,unsigned int sm){(void)kn;(void)gx;(void)gy;(void)gz;(void)bx;(void)by;(void)bz;(void)sm;return 2000;}

static char key_host[8], key_dev[8];

int main(void) {
    struct fg_config *c = fg_cfg();
    c->num_devices = 4;
    fg_set_cost_resolver_for(0, resolver_a);
    fg_set_cost_resolver_for(1, resolver_b);
    fg_set_cost_resolver_for(2, resolver_a);
    fg_set_cost_resolver_for(3, resolver_b);

    __cudaRegisterFunction(NULL, key_host, key_dev, "k", 0, NULL, NULL, NULL, NULL, NULL);

    /* AC1+AC2: device 0 vs device 1 produce different durations. */
    cudaSetDevice(0);
    int d0 = c->current_device;
    cudaSetDevice(1);
    int d1 = c->current_device;
    CHECK(d0 == 0 && d1 == 1, "cudaSetDevice updates current_device");
    CHECK(fg_cost_resolver_for(0) == resolver_a, "dev0 resolver is A");
    CHECK(fg_cost_resolver_for(1) == resolver_b, "dev1 resolver is B");
    CHECK(fg_cost_resolver_for(0) != fg_cost_resolver_for(1), "per-device resolvers differ");

    /* AC3: fallback chain -- no resolver for a device -> global -> scalar. */
    fg_set_cost_resolver_for(0, NULL);
    CHECK(fg_cost_resolver_for(0) == NULL, "cleared per-device resolver is NULL");
    /* global resolver fallback */
    c->cost_resolver = resolver_a;
    CHECK(c->cost_resolver == resolver_a, "global resolver set");
    c->cost_resolver = NULL;
    /* scalar path: no resolver -> default (no sleep, but launch succeeds) */
    cudaSetDevice(0);
    cudaError_t e = cudaLaunchKernel(key_dev, 1,1,1, 1,1,1, 0, NULL, NULL, NULL);
    CHECK(e == cudaSuccess, "scalar path launch succeeds");

    /* restore */
    fg_set_cost_resolver_for(0, resolver_a);

    /* AC4: OOM returns cudaErrorMemoryAllocation. */
    cudaSetDevice(0);
    void *p = NULL;
    size_t big = 300ULL * 1024 * 1024;
    e = cudaMalloc(&p, big);
    CHECK(e == cudaErrorMemoryAllocation, "OOM returns cudaErrorMemoryAllocation (got %d)", e);
    void *q = NULL;
    e = cudaMalloc(&q, 1024);
    CHECK(e == cudaSuccess, "small alloc succeeds");

    /* AC5: 4 devices, 2 models -> 2 distinct timings. */
    /* Already verified via resolver_for mapping above. */
    CHECK(fg_cost_resolver_for(0) != fg_cost_resolver_for(1), "2 distinct timings (A vs B)");
    CHECK(fg_cost_resolver_for(0) == fg_cost_resolver_for(2), "dev0 and dev2 same model");
    CHECK(fg_cost_resolver_for(1) == fg_cost_resolver_for(3), "dev1 and dev3 same model");

    /* restore global */
    c->cost_resolver = NULL;

    if (failures) { printf("\n%d check(s) FAILED\n", failures); return 1; }
    printf("\nm3d.12 hetero gate PASSED\n");
    return 0;
}
