/*
 * fakegpu_nccl.c — NCCL API interposer.
 *
 * Hooks NCCL symbols; world size defaults to FAKEGPU_NUM_DEVICES, rank to 0
 * (single-process multi-device simulation) unless FAKEGPU_NCCL_RANK is set.
 * Collective output buffers are filled with random bytes and latency is applied
 * based on size (via the bandwidth or fixed memcpy model).
 */
#include "fakegpu.h"
#include "fakegpu_nccl.h"
#include "nccl.h"
#include "fakegpu_topology.h"
#include "fakegpu_fabric.h"
#include <math.h>

/* Named refusal: model has no measured P for this (n, S); not a duration. */

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

/* ---- world size / rank resolution --------------------------------------- */

/* ---- world size / rank resolution (per-comm where available) ------------- */

static int fg_nccl_rank_from_env(void)
{
    const char *s = getenv("FAKEGPU_NCCL_RANK");
    if (s && *s) {
        int r = (int)strtol(s, NULL, 10);
        if (r < 0) r = 0;
        return r;
    }
    return 0;
}

static ncclComm_t fg_nccl_fake_comm(int nrank, int rank)
{
    int n = nrank > 0 ? nrank : (int)fg_cfg()->num_devices;
    if (n < 1) n = 1;
    struct fg_nccl_comm *c = (struct fg_nccl_comm *)malloc(sizeof(*c));
    if (!c) return NULL;
    c->nrank = n;
    c->rank  = rank;
    return (ncclComm_t)c;
}

/* ---- comm lifecycle ------------------------------------------------------ */

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nRanks, int rank,
                              unsigned long long commId)
{
    if (!comm) return ncclErrorInvalidArgument;
    (void)commId;
    /* nRanks defaults to FAKEGPU_NUM_DEVICES when <=0 (caller didn't specify) */
    int n = nRanks > 0 ? nRanks : fg_cfg()->num_devices;
    *comm = fg_nccl_fake_comm(n, rank);
    return *comm ? ncclSuccess : ncclSystemError;
}

ncclResult_t ncclCommInit32(ncclComm_t *comm, int nRanks, int rank,
                            unsigned long long commId)
{
    return ncclCommInitRank(comm, nRanks, rank, commId);
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t *comm, int nRanks, int rank,
                                        unsigned long long commId)
{
    return ncclCommInitRank(comm, nRanks, rank, commId);
}

ncclResult_t ncclCommInitRankScalable2(ncclComm_t *comm, int nranks, int rank,
                                       unsigned long long commId, const int *topo)
{
    (void)topo;
    return ncclCommInitRank(comm, nranks, rank, commId);
}

ncclResult_t ncclCommDestroy(ncclComm_t comm)
{
    free(comm);  /* fg_nccl_fake_comm malloc'd the struct */
    return ncclSuccess;
}

ncclResult_t ncclCommDestroy_Handle(ncclComm_t comm)
{
    return ncclCommDestroy(comm);
}

ncclResult_t ncclCommFinalize(ncclComm_t comm)
{
    (void)comm;
    return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm)
{
    (void)comm;
    return ncclSuccess;
}

ncclResult_t ncclCommGetRank(ncclComm_t comm, int *rank)
{
    if (!rank) return ncclErrorInvalidArgument;
    *rank = comm ? ((struct fg_nccl_comm *)comm)->rank : fg_nccl_rank_from_env();
    return ncclSuccess;
}

ncclResult_t ncclCommGetNrThreads(int *nThreads, ncclComm_t comm)
{
    if (!nThreads) return ncclErrorInvalidArgument;
    *nThreads = 1;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *nThreads)
{
    if (!nThreads) return ncclErrorInvalidArgument;
    *nThreads = comm ? ((struct fg_nccl_comm *)comm)->nrank : 1;
    return ncclSuccess;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t *newcomm)
{
    (void)color; (void)key;
    if (!newcomm) return ncclErrorInvalidArgument;
    int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : (int)fg_cfg()->num_devices;
    int rk = comm ? ((struct fg_nccl_comm *)comm)->rank  : 0;
    *newcomm = fg_nccl_fake_comm(nr, rk);
    return *newcomm ? ncclSuccess : ncclSystemError;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, int *errorCode)
{
    (void)comm;
    if (!errorCode) return ncclErrorInvalidArgument;
    *errorCode = 0;
    return ncclSuccess;
}

/* ---- group --------------------------------------------------------------- */

ncclResult_t ncclGroupStart(void)
{
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd(void)
{
    return ncclSuccess;
}

/* ---- point-to-point ------------------------------------------------------ */

ncclResult_t ncclSend(const void *sendbuff, size_t count, int datatype,
                      int peer, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)datatype; (void)peer; (void)comm; (void)stream;
    /* model a tiny latency proportional to count */
    fg_simulate_memcpy(count);
    return ncclSuccess;
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, int datatype,
                      int peer, ncclComm_t comm, void *stream)
{
    (void)datatype; (void)peer; (void)comm; (void)stream;
    fg_simulate_memcpy(count);
    if (recvbuff && count) fg_fill(recvbuff, count);
    return ncclSuccess;
}

ncclResult_t ncclSendRecv(void *recvbuf, size_t rcount, int rtype, int recvFrom,
                          const void *sendbuf, size_t scount, int stype,
                          int sendTo, ncclComm_t comm, void *stream)
{
    (void)sendbuf; (void)stype; (void)sendTo; (void)comm; (void)stream;
    fg_simulate_memcpy(rcount > scount ? rcount : scount);
    if (recvbuf && rcount) {
        if (rcount) fg_fill(recvbuf, rcount);
    }
    return ncclSuccess;
}

static fg_topo_t *fg_topo_cache = NULL;
static char fg_topo_path[4096] = "";

static fg_topo_t *fg_nccl_topo(void) {
    const char *p = getenv("GPUSIM_TOPOLOGY");
    if (!p || !*p) return NULL;
    if (fg_topo_cache && strcmp(p, fg_topo_path) == 0) return fg_topo_cache;
    char err[256];
    fg_topo_t *t = fg_topo_load(p, err, sizeof(err));
    if (!t) {
        fprintf(stderr, "fakegpu: corrupted topology: %s\n", err);
        exit(1);
    }
    if (fg_topo_cache) fg_topo_free(fg_topo_cache);
    fg_topo_cache = t;
    snprintf(fg_topo_path, sizeof(fg_topo_path), "%s", p);
    return t;
}

static size_t fg_dtype_size(int datatype) {
    switch (datatype) {
    case ncclInt8: return 1;
    case ncclFloat16: return 2;
    case ncclInt32: return 4;
    case ncclFloat32: return 4;
    case ncclInt64: return 8;
    case ncclFloat64: return 8;
    default: return 1;
    }
}

static const char *fg_dtype_name(int datatype) {
    switch (datatype) {
    case ncclInt8: return "int8";
    case ncclUint8: return "uint8";
    case ncclFloat16: return "float16";
    case ncclInt32: return "int32";
    case ncclFloat32: return "float32";
    case ncclInt64: return "int64";
    case ncclFloat64: return "float64";
    default: return "int8";
    }
}

static int fg_shortest_path(const fg_topo_t *t, int src, int dst, int *out, int *nout) {
    if (src == dst) { *nout = 0; return 0; }
    int n = t->n_dev;
    double *dist = calloc(n, sizeof(double));
    int *prev = calloc(n, sizeof(int));
    int *prevlink = calloc(n, sizeof(int));
    char *seen = calloc(n, 1);
    for (int i = 0; i < n; i++) { dist[i] = 1e18; prev[i] = -1; prevlink[i] = -1; }
    dist[src] = 0;
    int *la = calloc(t->n_links, sizeof(int));
    int *lb = calloc(t->n_links, sizeof(int));
    for (int li = 0; li < t->n_links; li++) {
        la[li] = fg_topo_dev_index(t, t->links[li].a_node, t->links[li].a_dev);
        lb[li] = fg_topo_dev_index(t, t->links[li].b_node, t->links[li].b_dev);
    }
    for (int iter = 0; iter < n; iter++) {
        int u = -1; double best = 1e18;
        for (int i = 0; i < n; i++) if (!seen[i] && dist[i] < best) { best = dist[i]; u = i; }
        if (u == -1) break;
        seen[u] = 1;
        if (u == dst) break;
        for (int li = 0; li < t->n_links; li++) {
            int v = -1;
            if (la[li] == u) v = lb[li];
            else if (lb[li] == u) v = la[li];
            if (v < 0 || seen[v]) continue;
            double w = t->links[li].latency_us;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                prev[v] = u;
                prevlink[v] = li;
            }
        }
    }
    int ok = (dist[dst] < 1e17);
    if (ok) {
        int cnt = 0;
        int cur = dst;
        while (cur != src) {
            out[cnt++] = prevlink[cur];
            cur = prev[cur];
        }
        for (int i = 0; i < cnt/2; i++) {
            int tmp = out[i]; out[i] = out[cnt-1-i]; out[cnt-1-i] = tmp;
        }
        *nout = cnt;
    }
    free(dist); free(prev); free(prevlink); free(seen);
    free(la); free(lb);
    return ok ? 0 : -1;
}

double fg_nccl_predict(const fg_topo_t *t, const char *algo, size_t count, int datatype, int nranks, int *bottleneck_link, char *regime, size_t regimesz) {
    if (nranks <= 1) { *bottleneck_link = -1; snprintf(regime, regimesz, "bandwidth"); return 0.0; }
    int n = nranks;
    int *rank_dev = calloc(n, sizeof(int));
    fg_topo_rank_t *sorted = malloc(t->n_ranks * sizeof(fg_topo_rank_t));
    memcpy(sorted, t->ranks, t->n_ranks * sizeof(fg_topo_rank_t));
    for (int i = 0; i < t->n_ranks; i++)
        for (int j = i+1; j < t->n_ranks; j++)
            if (sorted[j].rank < sorted[i].rank) {
                fg_topo_rank_t tmp = sorted[i]; sorted[i] = sorted[j]; sorted[j] = tmp;
            }
    for (int i = 0; i < n; i++) {
        if (i < t->n_ranks) {
            rank_dev[i] = fg_topo_dev_index(t, sorted[i].node_idx, sorted[i].dev_id);
        } else {
            rank_dev[i] = (i < t->n_dev) ? i : 0;
        }
        if (rank_dev[i] < 0) rank_dev[i] = 0;
    }
    free(sorted);

    double dtype_sz = (double)fg_dtype_size(datatype);
    double S = (double)count * dtype_sz;

    int maxedges = n * n;
    int (*edges)[2] = calloc(maxedges, sizeof(*edges));
    int nedges = 0;
    int is_tree = (strcmp(algo, "tree_bcast") == 0);
    if (is_tree) {
        for (int i = 1; i < n; i++) {
            edges[nedges][0] = (i-1)/2;
            edges[nedges][1] = i;
            nedges++;
        }
    } else {
        for (int i = 0; i < n; i++) {
            edges[nedges][0] = i;
            edges[nedges][1] = (i+1) % n;
            nedges++;
        }
    }

    double *edge_bw = calloc(nedges, sizeof(double));
    double *edge_lat = calloc(nedges, sizeof(double));
    int (*edge_paths)[64] = calloc(nedges, sizeof(*edge_paths));
    int *edge_pathlen = calloc(nedges, sizeof(int));

    for (int e = 0; e < nedges; e++) {
        int u = rank_dev[edges[e][0]];
        int v = rank_dev[edges[e][1]];
        int path[64];
        int plen = 0;
        if (u == v) { plen = 0; }
        else {
            if (fg_shortest_path(t, u, v, path, &plen) != 0) {
                plen = -1;
            }
        }
        if (plen < 0) {
            edge_bw[e] = 0;
            edge_lat[e] = 0;
            edge_pathlen[e] = 0;
        } else {
            edge_pathlen[e] = plen;
            for (int k = 0; k < plen; k++) edge_paths[e][k] = path[k];
            if (plen == 0) {
                edge_bw[e] = 1e9;
                edge_lat[e] = 0;
            } else {
                double bw = 1e9;
                double lat = 0;
                for (int k = 0; k < plen; k++) {
                    int li = path[k];
                    double ebw = t->links[li].eff_bw_gbs;
                    if (ebw < bw) bw = ebw;
                    lat += t->links[li].latency_us;
                }
                edge_bw[e] = bw;
                edge_lat[e] = lat;
            }
        }
    }

    int worst = 0;
    for (int e = 1; e < nedges; e++) {
        if (edge_bw[e] < edge_bw[worst] || (edge_bw[e] == edge_bw[worst] && edge_lat[e] > edge_lat[worst])) {
            worst = e;
        }
    }

    double B_gbps = edge_bw[worst];
    if (B_gbps <= 0 || B_gbps > 1e8) B_gbps = 1e9;
    double alpha_us = edge_lat[worst];
    double B_Bps = B_gbps * 1e9 / 8.0;
    double alpha_s = alpha_us * 1e-6;

    double dur, bw_term, lat_term;
    if (is_tree) {
        double logn = log2((double)n);
        if (n <= 1) logn = 1;
        int P_tree = (int)ceil((double)S / (double)FG_FAB_MAX_PAYLOAD);
        if (P_tree < 1) P_tree = 1;
        bw_term = logn * S / B_Bps;
        lat_term = logn * alpha_s * (double)P_tree;
        dur = bw_term + lat_term;
        /* fault: loss factor (gpu-sim-5zv) - use first link's loss_ppm */
        if (t->n_links > 0 && t->links[0].loss_ppm >= 1000000) {
            *bottleneck_link = -1;
            if (regime && regimesz) snprintf(regime, regimesz, "refused");
            return FG_NCCL_REFUSED;
        }
        if (t->n_links > 0 && t->links[0].loss_ppm > 0 && t->links[0].loss_ppm < 1000000) {
            double _p = t->links[0].loss_ppm / 1e6;
            dur = bw_term * (1.0 / (1.0 - _p)) + lat_term;
        }
    } else {
        double bf, lf;
        /* gpu-sim-ggh: n=2 special case removed. bf=2*(n-1)/n, so n=2 -> 1.0
           (matches eut committed traces: 1.00 at every clean size). n>=3
           unchanged. Latency factor 2*(n-1) for ALL n (ring hop count). Mirrors
           epic_cluster/link_model.py; asserted by C/Python agreement test. */
        double chunk = (double)(n - 1) / (double)n;  /* ggh: drop n==2 special case; n=2 -> 1/2, bf=1.0 */
        double hops  = (double)(n - 1);  /* 2*(n-1) latency: ring hop count, all n */
        if (strcmp(algo, "ring_allreduce") == 0) { bf = 2.0*chunk; lf = 2.0*hops; }
        else if (strcmp(algo, "ring_allgather") == 0 || strcmp(algo, "ring_reduce_scatter") == 0 || strcmp(algo, "ring_alltoall") == 0) { bf = (double)(n-1)/n; lf = (double)n; }
        else { bf = 1.0; lf = 1.0; }
        /* pkb: pieces per phase P comes from the MEASURED phase-payload table below,
           not from a cap-derived formula (n>=3 refusal: a cap-derived P would be a
           derivation, and experiment gpu-sim-6kp measured these values). The old
           comment here claimed P = ceil(chunk(n)*S / FG_FAB_MAX_PAYLOAD), which the
           code has not done since the table landed -- corrected by gpu-sim-ggh review. */
        int P;
        if (n == 2 && S <= 4096) P = 1;           /* phase-payload table */
        else if (n == 2 && S <= 262144) P = 4;    /* phase-payload table */
        else if (n == 2 && S <= 2097152) P = 8;   /* phase-payload table */
        else {
            /* outside measured range: refuse rather than derive from cap */
            *bottleneck_link = -1;
            if (regime && regimesz) snprintf(regime, regimesz, "refused");
            return FG_NCCL_REFUSED;
        }
        bw_term = bf * S / B_Bps;
        lat_term = lf * alpha_s * (double)P;
        dur = bw_term + lat_term;
        /* fault: expected retransmit factor under loss (gpu-sim-5zv) */
        {
            int _loss_ppm = 0;
            if (worst < nedges && edge_pathlen[worst] > 0) {
                int _li = edge_paths[worst][0];
                if (_li >= 0 && _li < t->n_links) _loss_ppm = (int)t->links[_li].loss_ppm;
            }
            if (_loss_ppm >= 1000000) {
                *bottleneck_link = -1;
                if (regime && regimesz) snprintf(regime, regimesz, "refused");
                return FG_NCCL_REFUSED;
            }
            if (_loss_ppm > 0 && _loss_ppm < 1000000) {
                double _p = _loss_ppm / 1e6;
                dur = bw_term * (1.0 / (1.0 - _p)) + lat_term;
            }
        }
    }

    int b_link = -1;
    int wpath_len = edge_pathlen[worst];
    if (wpath_len > 0) {
        b_link = edge_paths[worst][0];
        for (int k = 1; k < wpath_len; k++) {
            int li = edge_paths[worst][k];
            if (t->links[li].eff_bw_gbs < t->links[b_link].eff_bw_gbs) b_link = li;
            else if (t->links[li].eff_bw_gbs == t->links[b_link].eff_bw_gbs && t->links[li].latency_us > t->links[b_link].latency_us) b_link = li;
        }
    }

    *bottleneck_link = b_link;
    if (lat_term > bw_term) snprintf(regime, regimesz, "latency");
    else snprintf(regime, regimesz, "bandwidth");

    free(rank_dev); free(edge_bw); free(edge_lat); free(edge_paths); free(edge_pathlen); free(edges);
    return dur;
}

static void fg_nccl_trace(const char *op, int rank, int nranks, size_t count, const char *dtype, size_t bytes, const char *algo, double predicted, int bottleneck, const char *regime) {
    const char *p = getenv("GPUSIM_TRACE");
    char path[4096];
    if (p && *p) snprintf(path, sizeof(path), "%s", p);
    else snprintf(path, sizeof(path), "gpusim_trace.jsonl");
    FILE *f = fopen(path, "a");
    if (!f) return;
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    long long ts_ns = (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
    if (predicted == FG_NCCL_REFUSED) {
        fprintf(f, "{\"ts_ns\":%lld,\"rank\":%d,\"nranks\":%d,\"op\":\"%s\",\"count\":%zu,\"dtype\":\"%s\",\"bytes\":%zu,\"algorithm\":\"%s\",\"predicted_s\":null,\"refused\":true,\"bottleneck_link\":", ts_ns, rank, nranks, op, count, dtype, bytes, algo);
    } else {
        fprintf(f, "{\"ts_ns\":%lld,\"rank\":%d,\"nranks\":%d,\"op\":\"%s\",\"count\":%zu,\"dtype\":\"%s\",\"bytes\":%zu,\"algorithm\":\"%s\",\"predicted_s\":%.9g,\"bottleneck_link\":", ts_ns, rank, nranks, op, count, dtype, bytes, algo, predicted);
    }
    if (bottleneck >= 0) fprintf(f, "%d", bottleneck); else fprintf(f, "null");
    fprintf(f, ",\"regime\":\"%s\"}\n", regime);
    fflush(f);
    fclose(f);
}

/* ---- collectives --------------------------------------------------------- */

static void fg_collective_out(void *recvbuff, size_t count, int datatype,
                            ncclComm_t comm)
{
    size_t bytes;
    switch (datatype) {
    case ncclInt8:    bytes = count * 1; break;
    case ncclUint8:   bytes = count * 1; break;
    case ncclInt32:   bytes = count * 4; break;
    case ncclUint32:  bytes = count * 4; break;
    case ncclInt64:   bytes = count * 8; break;
    case ncclUint64:  bytes = count * 8; break;
    case ncclFloat32: bytes = count * 4; break;
    case ncclFloat64: bytes = count * 8; break;
    case ncclFloat16: bytes = count * 2; break;
    default:          bytes = count; break;
    }
    /* scale by log2(nrank) to model fan-in/fan-out cost */
    int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : 1;
    int lvl = 0;
    int tmp = nr;
    while (tmp > 1) { lvl++; tmp >>= 1; }
    if (lvl > 0) bytes = bytes + bytes * (size_t)lvl / 2;
    fg_simulate_memcpy(bytes);
    if (recvbuff && bytes) fg_fill(recvbuff, bytes);
}

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)op; (void)stream;
    fg_topo_t *t = fg_nccl_topo();
    if (t) {
        int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : (int)fg_cfg()->num_devices;
        int rk = comm ? ((struct fg_nccl_comm *)comm)->rank : 0;
        int bl = -1; char regime[16];
        double pred = fg_nccl_predict(t, "ring_allreduce", count, datatype, nr, &bl, regime, sizeof(regime));
        if (pred != FG_NCCL_REFUSED) fg_sleep_ns((long)(pred * 1e9));
        size_t bytes = count * fg_dtype_size(datatype);
        if (recvbuff && bytes) fg_fill(recvbuff, bytes);
        fg_nccl_trace("allreduce", rk, nr, count, fg_dtype_name(datatype), bytes, "ring_allreduce", pred, bl, regime);
        return ncclSuccess;
    }
    fg_collective_out(recvbuff, count, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclAllReduce_v2(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int op, ncclComm_t comm, void *stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int root, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)root; (void)stream;
    fg_topo_t *t = fg_nccl_topo();
    if (t) {
        int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : (int)fg_cfg()->num_devices;
        int rk = comm ? ((struct fg_nccl_comm *)comm)->rank : 0;
        int bl = -1; char regime[16];
        double pred = fg_nccl_predict(t, "tree_bcast", count, datatype, nr, &bl, regime, sizeof(regime));
        if (pred != FG_NCCL_REFUSED) fg_sleep_ns((long)(pred * 1e9));
        size_t bytes = count * fg_dtype_size(datatype);
        if (recvbuff && bytes) fg_fill(recvbuff, bytes);
        fg_nccl_trace("broadcast", rk, nr, count, fg_dtype_name(datatype), bytes, "tree_bcast", pred, bl, regime);
        return ncclSuccess;
    }
    fg_collective_out(recvbuff, count, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclBroadcast_v2(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int root, ncclComm_t comm, void *stream)
{
    return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t ncclBroadcast_v3(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int root, ncclComm_t comm, void *stream,
                              int priority)
{
    (void)priority;
    return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        int datatype, int op, int root, ncclComm_t comm,
                        void *stream)
{
    (void)sendbuff; (void)op; (void)root; (void)stream;
    fg_topo_t *t = fg_nccl_topo();
    if (t) {
        int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : (int)fg_cfg()->num_devices;
        int rk = comm ? ((struct fg_nccl_comm *)comm)->rank : 0;
        int bl = -1; char regime[16];
        double pred = fg_nccl_predict(t, "ring_allreduce", count, datatype, nr, &bl, regime, sizeof(regime));
        if (pred != FG_NCCL_REFUSED) fg_sleep_ns((long)(pred * 1e9));
        size_t bytes = count * fg_dtype_size(datatype);
        if (recvbuff && bytes) fg_fill(recvbuff, bytes);
        fg_nccl_trace("reduce", rk, nr, count, fg_dtype_name(datatype), bytes, "ring_allreduce", pred, bl, regime);
        return ncclSuccess;
    }
    fg_collective_out(recvbuff, count, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclReduce_v2(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, int root, ncclComm_t comm,
                           void *stream)
{
    return ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t ncclReduce_v3(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, int root, ncclComm_t comm,
                           void *stream, int priority)
{
    (void)priority;
    return ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)stream;
    fg_topo_t *t = fg_nccl_topo();
    if (t) {
        int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : (int)fg_cfg()->num_devices;
        int rk = comm ? ((struct fg_nccl_comm *)comm)->rank : 0;
        int bl = -1; char regime[16];
        double pred = fg_nccl_predict(t, "ring_allgather", count, datatype, nr, &bl, regime, sizeof(regime));
        if (pred != FG_NCCL_REFUSED) fg_sleep_ns((long)(pred * 1e9));
        size_t bytes = count * (size_t)nr * fg_dtype_size(datatype);
        if (recvbuff && bytes) fg_fill(recvbuff, bytes);
        fg_nccl_trace("allgather", rk, nr, count, fg_dtype_name(datatype), count * fg_dtype_size(datatype), "ring_allgather", pred, bl, regime);
        return ncclSuccess;
    }
    int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : 1;
    size_t full = count * (size_t)nr;
    fg_collective_out(recvbuff, full, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclAllGather_v2(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, ncclComm_t comm, void *stream)
{
    return ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                               const size_t *recvcount, int datatype, int op,
                               ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)op; (void)stream;
    size_t rc = recvcount ? recvcount[0] : 0;
    fg_topo_t *t = fg_nccl_topo();
    if (t) {
        int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : (int)fg_cfg()->num_devices;
        int rk = comm ? ((struct fg_nccl_comm *)comm)->rank : 0;
        int bl = -1; char regime[16];
        double pred = fg_nccl_predict(t, "ring_reduce_scatter", rc, datatype, nr, &bl, regime, sizeof(regime));
        if (pred != FG_NCCL_REFUSED) fg_sleep_ns((long)(pred * 1e9));
        size_t bytes = rc * fg_dtype_size(datatype);
        if (recvbuff && bytes) fg_fill(recvbuff, bytes);
        fg_nccl_trace("reducescatter", rk, nr, rc, fg_dtype_name(datatype), bytes, "ring_reduce_scatter", pred, bl, regime);
        return ncclSuccess;
    }
    fg_collective_out(recvbuff, rc, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclAllToAll(const void *sendbuff, void *recvbuff, size_t count,
                          int datatype, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)stream;
    fg_topo_t *t = fg_nccl_topo();
    if (t) {
        int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : (int)fg_cfg()->num_devices;
        int rk = comm ? ((struct fg_nccl_comm *)comm)->rank : 0;
        int bl = -1; char regime[16];
        double pred = fg_nccl_predict(t, "ring_alltoall", count, datatype, nr, &bl, regime, sizeof(regime));
        if (pred != FG_NCCL_REFUSED) fg_sleep_ns((long)(pred * 1e9));
        size_t bytes = count * fg_dtype_size(datatype);
        if (recvbuff && bytes) fg_fill(recvbuff, bytes);
        fg_nccl_trace("alltoall", rk, nr, count, fg_dtype_name(datatype), bytes, "ring_alltoall", pred, bl, regime);
        return ncclSuccess;
    }
    fg_collective_out(recvbuff, count, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclAllToAll_v2(const void *sendbuff, void *recvbuff, size_t count,
                             int datatype, ncclComm_t comm, void *stream)
{
    return ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

/* ---- op / version helpers ------------------------------------------------ */

ncclResult_t ncclRedOpCreate(ncclComm_t comm, int op, void *func, int datatype)
{
    (void)comm; (void)func; (void)datatype;
    return (ncclResult_t)op;
}

ncclResult_t ncclRedOpDestroy(int op)
{
    (void)op;
    return ncclSuccess;
}

ncclResult_t ncclGetVersion(int *version)
{
    if (!version) return ncclErrorInvalidArgument;
    *version = 212000; /* fake NCCL version 2.12 */
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(unsigned long long *commId)
{
    if (!commId) return ncclErrorInvalidArgument;
    *commId = fg_rng_next();
    return ncclSuccess;
}

ncclResult_t ncclSetThreadPerBlock(int thrPerBlock)
{
    (void)thrPerBlock;
    return ncclSuccess;
}

const char *ncclGetErrorString(ncclResult_t code)
{
    switch (code) {
    case ncclSuccess:              return "no error";
    case ncclErrorInvalidNumber:   return "invalid number";
    case ncclErrorInvalidArgument: return "invalid argument";
    default:                       return "unknown error";
    }
}
