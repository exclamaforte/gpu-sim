#define _GNU_SOURCE
#include "fakegpu_topology.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER_SIZE 20
#define DEVICE_SIZE 88
#define LINK_SIZE 120
#define RANK_SIZE 12
#define MAGIC 0x47535043

static void seterr(char *buf, size_t sz, const char *msg) {
    if (buf && sz) snprintf(buf, sz, "%s", msg);
}

fg_topo_t *fg_topo_load(const char *path, char *errbuf, size_t errsz) {
    if (!path) { seterr(errbuf, errsz, "null path"); return NULL; }
    FILE *f = fopen(path, "rb");
    if (!f) { seterr(errbuf, errsz, "cannot open topology file"); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < HEADER_SIZE) { fclose(f); seterr(errbuf, errsz, "topology file too small"); return NULL; }
    unsigned char *data = malloc(sz);
    if (!data) { fclose(f); seterr(errbuf, errsz, "out of memory"); return NULL; }
    if (fread(data, 1, sz, f) != (size_t)sz) { fclose(f); free(data); seterr(errbuf, errsz, "short read"); return NULL; }
    fclose(f);

    uint32_t magic, version, n_dev, n_links, n_ranks;
    memcpy(&magic, data, 4);
    memcpy(&version, data+4, 4);
    memcpy(&n_dev, data+8, 4);
    memcpy(&n_links, data+12, 4);
    memcpy(&n_ranks, data+16, 4);

    if (magic != MAGIC) { free(data); seterr(errbuf, errsz, "bad magic"); return NULL; }
    if (version != 1) { free(data); seterr(errbuf, errsz, "bad version"); return NULL; }
    long expected = HEADER_SIZE + (long)n_dev*DEVICE_SIZE + (long)n_links*LINK_SIZE + (long)n_ranks*RANK_SIZE;
    if (expected != sz) { free(data); seterr(errbuf, errsz, "size mismatch"); return NULL; }

    fg_topo_t *t = calloc(1, sizeof(*t));
    t->magic = magic;
    t->version = version;
    t->n_dev = n_dev;
    t->n_links = n_links;
    t->n_ranks = n_ranks;
    t->devs = calloc(n_dev ? n_dev : 1, sizeof(fg_topo_dev_t));
    t->links = calloc(n_links ? n_links : 1, sizeof(fg_topo_link_t));
    t->ranks = calloc(n_ranks ? n_ranks : 1, sizeof(fg_topo_rank_t));

    size_t off = HEADER_SIZE;
    for (int i = 0; i < (int)n_dev; i++) {
        fg_topo_dev_t *d = &t->devs[i];
        uint32_t node_idx, dev_id, sm_count, clock_mhz;
        uint64_t mem_bytes;
        double mem_bw;
        memcpy(&node_idx, data+off, 4); off += 4;
        memcpy(&dev_id, data+off, 4); off += 4;
        memcpy(d->model_name, data+off, 32); off += 32;
        memcpy(&sm_count, data+off, 4); off += 4;
        memcpy(&clock_mhz, data+off, 4); off += 4;
        memcpy(&mem_bytes, data+off, 8); off += 8;
        memcpy(&mem_bw, data+off, 8); off += 8;
        memcpy(d->arch, data+off, 16); off += 16;
        off += 8; /* padding */
        d->node_idx = node_idx;
        d->dev_id = dev_id;
        d->sm_count = sm_count;
        d->clock_mhz = clock_mhz;
        d->mem_bytes = mem_bytes;
        d->mem_bw_gbs = mem_bw;
    }
    for (int i = 0; i < (int)n_links; i++) {
        fg_topo_link_t *l = &t->links[i];
        uint32_t a_node, a_dev, b_node, b_dev, mtu, loss;
        double bw, lat, jitter, eff;
        char sg[32];
        memcpy(&a_node, data+off, 4); off += 4;
        memcpy(&a_dev, data+off, 4); off += 4;
        memcpy(&b_node, data+off, 4); off += 4;
        memcpy(&b_dev, data+off, 4); off += 4;
        memcpy(&bw, data+off, 8); off += 8;
        memcpy(&lat, data+off, 8); off += 8;
        memcpy(&mtu, data+off, 4); off += 4;
        memcpy(&loss, data+off, 4); off += 4;
        memcpy(&jitter, data+off, 8); off += 8;
        memcpy(sg, data+off, 32); off += 32;
        memcpy(&eff, data+off, 8); off += 8;
        off += 32; /* padding */
        l->a_node = a_node; l->a_dev = a_dev;
        l->b_node = b_node; l->b_dev = b_dev;
        l->bw_gbps = bw; l->latency_us = lat;
        l->mtu = mtu; l->loss_ppm = loss;
        l->jitter_us = jitter;
        memcpy(l->shared_group, sg, 32);
        l->eff_bw_gbs = eff;
    }
    for (int i = 0; i < (int)n_ranks; i++) {
        uint32_t rank, node_idx, dev_id;
        memcpy(&rank, data+off, 4); off += 4;
        memcpy(&node_idx, data+off, 4); off += 4;
        memcpy(&dev_id, data+off, 4); off += 4;
        t->ranks[i].rank = rank;
        t->ranks[i].node_idx = node_idx;
        t->ranks[i].dev_id = dev_id;
    }
    free(data);
    return t;
}

void fg_topo_free(fg_topo_t *t) {
    if (!t) return;
    free(t->devs);
    free(t->links);
    free(t->ranks);
    free(t);
}

int fg_topo_dev_index(const fg_topo_t *t, uint32_t node_idx, uint32_t dev_id) {
    if (!t) return -1;
    for (int i = 0; i < t->n_dev; i++) {
        if (t->devs[i].node_idx == node_idx && t->devs[i].dev_id == dev_id) return i;
    }
    return -1;
}