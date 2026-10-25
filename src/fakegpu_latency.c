/*
 * fakegpu_latency.c — delay injection.
 *
 * Two modes per operation:
 *  - fixed latency (FAKEGPU_*_LATENCY_*): always sleeps that duration.
 *  - bandwidth scaling (FAKEGPU_BANDWIDTH_GBPS): time = bytes / bandwidth,
 *    used when present, otherwise fixed latency.
 *
 * When all relevant env vars are 0/unset we take a fast path (no sleeping) so
 * CI can stay speedy by default.
 */
#include "fakegpu.h"
#include "fakegpu_topology.h"

#include <errno.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

static long g_have_any_latency = 0; /* set once a nonzero latency is observed */

void fg_sleep_ns(long ns)
{
    if (ns <= 0) return;
    struct timespec ts;
    ts.tv_sec  = ns / 1000000000L;
    ts.tv_nsec = ns % 1000000000L;
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {
        /* loop: nanosleep may be interrupted by a signal */
    }
}

void fg_simulate_memcpy(size_t bytes)
{
    const struct fg_config *c = fg_cfg();
    long ns = c->memcpy_latency_ns;
    if (c->bandwidth_gbps > 0.0 && bytes > 0) {
        /* t = bytes / (GBps * 1e9 / 8) = bytes * 8e9 / GBps  seconds */
        double ns_d = (double)bytes * 8.0 / (c->bandwidth_gbps * 1e9);
        long bw_ns = (long)ns_d;
        if (bw_ns > ns) ns = bw_ns;
    }
    if (ns > 0) g_have_any_latency = 1;
    fg_sleep_ns(ns);
}

void fg_simulate_kernel(void)
{
    fg_simulate_kernel_resolved(NULL, 0, 0, 0, 0, 0, 0, 0);
}

void fg_simulate_kernel_resolved(const char *kernel_name,
                                 int gridX, int gridY, int gridZ,
                                 int blockX, int blockY, int blockZ,
                                 unsigned int sharedMem)
{
    const struct fg_config *c = fg_cfg();
    /* Per-device resolver (gpu-sim-m3d.12): device id from active device. */
    int dev = c->current_device;
    if (dev < 0 || dev >= c->num_devices) dev = 0;
    fg_cost_resolver_fn resolver = fg_cost_resolver_for(dev);
    if (!resolver) resolver = c->cost_resolver;
    /* Resolved path: the resolver owns the number. Its return flows only
     * into fg_sleep_ns -- there is no record, no artifact, no writeback
     * anywhere downstream (D_GPUSIM: consume, never produce). */
    if (resolver && kernel_name) {
        long ns = resolver(kernel_name, gridX, gridY, gridZ,
                           blockX, blockY, blockZ, sharedMem);
        if (ns > 0) g_have_any_latency = 1;
        fg_sleep_ns(ns);
        return;
    }
    /* Default path: today's scalar behaviour. NULL resolver (static
     * zero-init default) or an unresolvable site both land here; neither
     * consults nor disturbs any model state. */
    if (c->kernel_latency_ms > 0) {
        g_have_any_latency = 1;
        fg_sleep_ns((long)(c->kernel_latency_ms * 1000000.0));
        return;
    }
    /* gpu-sim-torch-harness: synthetic mode. With no model loaded, a run still
     * has to produce plausible per-kernel timing or a trace of it is empty.
     * FAKEGPU_LATENCY_MODE=random draws each kernel's duration uniformly from
     * [FAKEGPU_LATENCY_MIN_MS, FAKEGPU_LATENCY_MAX_MS]; the point is to give the
     * scheduler and the trace real structure to reason about before any
     * calibration exists. */
    {
        const char *mode = getenv("FAKEGPU_LATENCY_MODE");
        if (mode && strcmp(mode, "random") == 0) {
            double lo = 0.01, hi = 1.0;
            const char *smin = getenv("FAKEGPU_LATENCY_MIN_MS");
            const char *smax = getenv("FAKEGPU_LATENCY_MAX_MS");
            if (smin && *smin) lo = strtod(smin, NULL);
            if (smax && *smax) hi = strtod(smax, NULL);
            if (hi < lo) { double t = lo; lo = hi; hi = t; }
            unsigned long long r = fg_rng_next();
            double u = (double)(r % 1000000ULL) / 1000000.0;
            double ms = lo + u * (hi - lo);
            g_have_any_latency = 1;
            fg_sleep_ns((long)(ms * 1000000.0));
        }
    }
}

void fg_simulate_sync(void)
{
    const struct fg_config *c = fg_cfg();
    if (c->sync_latency_ms > 0) {
        g_have_any_latency = 1;
        fg_sleep_ns((long)(c->sync_latency_ms * 1000000.0));
    }
}

long fg_latency_active(void) { return g_have_any_latency; }
