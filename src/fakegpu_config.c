/*
 * fakegpu_config.c — environment-driven configuration + latency model.
 */
#include "fakegpu.h"
#include "fakegpu_topology.h"

#include <stdlib.h>
#include <string.h>

#define DEFAULT_NUM_DEVICES 1
#define DEFAULT_DEVICE_MEM_MB 256
#define U64(x) ((uint64_t)(x))

static struct fg_config g_cfg;
static fg_topo_t *g_topo = NULL;
static int g_init = 0;

static long parse_long(const char *s, long dflt)
{
    if (!s || !*s) return dflt;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    return (end == s) ? dflt : v;
}

static double parse_double(const char *s, double dflt)
{
    if (!s || !*s) return dflt;
    char *end = NULL;
    double v = strtod(s, &end);
    return (end == s) ? dflt : v;
}

void fg_config_reload(void)
{
    /* Load topology spec if available */
    const char *tp = getenv("GPUSIM_TOPOLOGY");
    if (tp && *tp) {
        char err[256];
        g_topo = fg_topo_load(tp, err, sizeof(err));
        if (g_topo && g_topo->n_dev > 0) {
            g_cfg.num_devices = g_topo->n_dev;
        } else {
            g_cfg.num_devices = (int) parse_long(getenv("FAKEGPU_NUM_DEVICES"), DEFAULT_NUM_DEVICES);
        }
    } else {
        g_cfg.num_devices = (int) parse_long(getenv("FAKEGPU_NUM_DEVICES"), DEFAULT_NUM_DEVICES);
    }
    if (g_cfg.num_devices < 1) g_cfg.num_devices = 1;
    g_cfg.device_mem_mb      = (size_t)parse_long(getenv("FAKEGPU_DEVICE_MEM_MB"),
                                                 DEFAULT_DEVICE_MEM_MB);
    /* memcpy latency: FAKEGPU_MEMCPY_LATENCY_US (microseconds) -> nanoseconds */
    g_cfg.memcpy_latency_ns  = parse_long(getenv("FAKEGPU_MEMCPY_LATENCY_US"), 0) * 1000L;
    /* bandwidth (GB/s): 0 disables bandwidth scaling (fixed latency only)  */
    g_cfg.bandwidth_gbps     = parse_double(getenv("FAKEGPU_BANDWIDTH_GBPS"), 0.0);
    /* fixed per-call latencies */
    g_cfg.kernel_latency_ms  = parse_long(getenv("FAKEGPU_KERNEL_LATENCY_MS"), 0);
    g_cfg.sync_latency_ms    = parse_long(getenv("FAKEGPU_DEVICE_SYNC_LATENCY_MS"), 0);
    g_cfg.hostfunc_latency_us= parse_long(getenv("FAKEGPU_HOST_FUNC_LATENCY_US"), 0);
    /* seed */
    const char *sseed = getenv("FAKEGPU_SEED");
    if (sseed && *sseed) {
        g_cfg.seed = (uint64_t)strtoull(sseed, NULL, 10);
    } else {
        /* default: time-based, set in fg_rng_init */
        g_cfg.seed = 0;
    }
    g_cfg.current_device = 0;
    g_init = 1;
}

struct fg_config *fg_cfg(void)
{
    if (!g_init) fg_config_reload();
    return &g_cfg;
}
