#ifndef FAKEGPU_TOPOLOGY_H
#define FAKEGPU_TOPOLOGY_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t node_idx;
    uint32_t dev_id;
    char model_name[32];
    uint32_t sm_count;
    uint32_t clock_mhz;
    uint64_t mem_bytes;
    double mem_bw_gbs;
    char arch[16];
} fg_topo_dev_t;

typedef struct {
    uint32_t a_node, a_dev;
    uint32_t b_node, b_dev;
    double bw_gbps;
    double latency_us;
    uint32_t mtu;
    uint32_t loss_ppm;
    double jitter_us;
    char shared_group[32];
    double eff_bw_gbs;
} fg_topo_link_t;

typedef struct {
    uint32_t rank;
    uint32_t node_idx;
    uint32_t dev_id;
} fg_topo_rank_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    int n_dev;
    int n_links;
    int n_ranks;
    fg_topo_dev_t *devs;
    fg_topo_link_t *links;
    fg_topo_rank_t *ranks;
} fg_topo_t;

#ifdef __cplusplus
/* C linkage guard: this header's functions are implemented in C
 * (src/fakegpu_topology.c). Without it, a C++ translation unit that
 * includes this header references them by MANGLED name and the resulting
 * shared object has an undefined symbol at load time. That happened on
 * 2026-08-22: the net plugin failed to load with
 *   undefined symbol: _Z12fg_topo_loadPKcPcm
 * and NCCL SILENTLY fell back to its built-in Socket/P2P transport, so a
 * run reported 'allreduce sum[0]=3.0 OK' over the WRONG transport. Consumers
 * must not have to remember a local extern "C" wrapper; the guard belongs
 * with the declarations. */
extern "C" {
#endif
/* Load a flat topology binary. Returns NULL on failure and sets errbuf. */
fg_topo_t *fg_topo_load(const char *path, char *errbuf, size_t errsz);
void fg_topo_free(fg_topo_t *t);

/* Map (node_idx, dev_id) -> device index in t->devs, or -1. */
int fg_topo_dev_index(const fg_topo_t *t, uint32_t node_idx, uint32_t dev_id);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif