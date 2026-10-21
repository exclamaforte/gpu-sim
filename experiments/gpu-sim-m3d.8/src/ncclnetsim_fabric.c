/* gpu-sim-m3d.8: ncclNet plugin over the shm fabric. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fakegpu_fabric.h"
#include "fakegpu_topology.h"
#include "nccl_net.h"

static fg_fab_t *FAB = NULL;
static fg_topo_t *TOPO = NULL;
/* 2b decider: THREE separate per-entry-point counters (never an aggregate) and
   a per-connection channel id. Counters are reported in the structured trace
   and in metrics.json. */
static unsigned long CNT_ISEND = 0, CNT_IRECV = 0, CNT_RECV_DONE = 0;

/* FNV-1a 64 of the payload: distinguishes genuine distinct transfers from one
   transfer recorded twice (same digest on the same channel = duplicate row). */
static unsigned long long fg_digest(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
static int CHAN_SEQ = 0;
static int MY_RANK = 0;
static int NRANKS = 1;
static pthread_mutex_t LOCK = PTHREAD_MUTEX_INITIALIZER;

#define CAP FG_FAB_MAX_PAYLOAD  /* must agree with fabric hard cap */
#if CAP != FG_FAB_MAX_PAYLOAD
#error "plugin CAP and fabric FG_FAB_MAX_PAYLOAD disagree"
#endif

typedef struct { int done; size_t size; void *data; int peer; int is_send; int chan_of_req; } req_t;

/* comm handles */
typedef struct { int peer; int local; int chan; } sim_comm_t;


static fg_topo_t *plugin_topo(void) {
    if (TOPO) return TOPO;
    const char *p = getenv("GPUSIM_TOPOLOGY");
    if (!p || !*p) return NULL;
    char err[256];
    fg_topo_t *t = fg_topo_load(p, err, sizeof(err));
    if (!t) {
        fprintf(stderr, "plugin: cannot load topology %s: %s\n", p, err);
        return NULL;
    }
    TOPO = t;
    return TOPO;
}

/* Resolve the link between src_rank and dst_rank in the loaded topology.
   Returns 0 on success and fills *link_id, *bw_gbps, *latency_us. */
static int plugin_resolve_link(int src_rank, int dst_rank, int *link_id,
                              double *bw_gbps, double *latency_us,
                              double *jitter_us, int *loss_ppm,
                              uint64_t *partition_after, double *timeout_us) {
    fg_topo_t *t = plugin_topo();
    if (!t) return -1;
    fg_topo_rank_t *sr = NULL, *dr = NULL;
    for (int i = 0; i < t->n_ranks; i++) {
        if (t->ranks[i].rank == (uint32_t)src_rank) sr = &t->ranks[i];
        if (t->ranks[i].rank == (uint32_t)dst_rank) dr = &t->ranks[i];
    }
    if (!sr || !dr) return -1;
    int si = fg_topo_dev_index(t, sr->node_idx, sr->dev_id);
    int di = fg_topo_dev_index(t, dr->node_idx, dr->dev_id);
    if (si < 0 || di < 0) return -1;
    for (int i = 0; i < t->n_links; i++) {
        int a = fg_topo_dev_index(t, t->links[i].a_node, t->links[i].a_dev);
        int b = fg_topo_dev_index(t, t->links[i].b_node, t->links[i].b_dev);
        if ((a == si && b == di) || (a == di && b == si)) {
            if (link_id) *link_id = i;
            double bw = t->links[i].eff_bw_gbs;
            if (bw <= 0.0) bw = t->links[i].bw_gbps;
            if (bw <= 0.0) return -2; /* zero bandwidth -> loud failure */
            if (bw_gbps) *bw_gbps = bw;
            if (latency_us) *latency_us = t->links[i].latency_us;
            if (jitter_us) *jitter_us = t->links[i].jitter_us;
            if (loss_ppm) *loss_ppm = t->links[i].loss_ppm;
            if (partition_after) *partition_after = t->links[i].partition_after;
            if (timeout_us) *timeout_us = t->links[i].timeout_us;
            return 0;
        }
    }
    return -1; /* missing link */
}

static void trace(const char *m, const char *extra) {
    const char *p = getenv("GPUSIM_TRACE");
    if (!p || !*p) return;
    FILE *f = fopen(p, "a");
    if (!f) return;
    /* rank on every line: both ranks append to one file, so without it the
       record cannot be split per rank and looks like missing calls. */
    fprintf(f, "rank=%d %s%s%s\n", MY_RANK, m, (extra && *extra) ? " " : "", extra ? extra : "");
    fflush(f);
    fclose(f);
}

/* v12 net */
static void *v12_ctx = NULL;

static ncclResult_t v12_init(void** ctx, uint64_t commId, ncclNetCommConfig_v12_t* config,
                            ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
    (void)config;(void)logFunction;(void)profFunction;
    const char *rid = getenv("GPUSIM_RUN_ID");
    if (!rid) rid = "m3d8";
    const char *r = getenv("GPUSIM_RANK");
    const char *n = getenv("GPUSIM_NRANKS");
    MY_RANK = r ? atoi(r) : 0;
    NRANKS = n ? atoi(n) : 1;
    uint64_t to = 5000000ULL; /* 5ms */
    const char *te = getenv("GPUSIM_TIMEOUT_NS");
    if (te) to = strtoull(te, NULL, 10);
    int err = 0;
    /* Attach may race the creator's segment materialization; retry on
       ETIMEDOUT rather than letting NCCL silently fall back to its built-in
       Socket transport. */
    for (int attempt = 0; attempt < 10 && !FAB; attempt++) {
        FAB = fg_fab_attach(rid, 1, MY_RANK, NRANKS, to, &err);
        if (!FAB && err == FG_FAB_ETIMEDOUT) usleep(200000);
    }
    if (!FAB) {
        fprintf(stderr, "fabric attach failed: %s\n", fg_fab_errstr(err));
        return ncclSystemError;
    }
    /* Per-link timing (gpu-sim-pkb): if a topology spec is supplied it must
       load and resolve; a missing/bad spec fails loudly here. */
    const char *tp = getenv("GPUSIM_TOPOLOGY");
    if (tp && *tp) {
        if (!plugin_topo()) {
            fprintf(stderr, "plugin: topology load failed for %s\n", tp);
            return ncclSystemError;
        }
    }
    if (ctx) *ctx = &v12_ctx;
    char buf[128]; snprintf(buf, sizeof(buf), "init commId=%llu rank=%d nranks=%d", (unsigned long long)commId, MY_RANK, NRANKS);
    trace("v12.init", buf);
    return ncclSuccess;
}

static ncclResult_t v12_devices(int* ndev) {
    if (ndev) *ndev = 1;
    trace("v12.devices", "");
    return ncclSuccess;
}

static ncclResult_t v12_getProperties(int dev, ncclNetProperties_v12_t* props) {
    if (props) {
        memset(props, 0, sizeof(*props));
        props->name = (char*)"sim";
        props->pciPath = (char*)"";
        props->guid = 0;
        props->ptrSupport = NCCL_PTR_HOST;
        props->regIsGlobal = 1;
        props->forceFlush = 0;
        props->speed = 100000;
        props->port = 0;
        props->latency = 1.0f;
        props->maxComms = 1<<20;
        props->maxRecvs = 1;
        props->netDeviceType = NCCL_NET_DEVICE_HOST;
        props->netDeviceVersion = 0;
        props->vProps.ndevs = 1;
        props->vProps.devs[0] = dev;
        props->maxP2pBytes = CAP;
        props->maxCollBytes = CAP;
        props->maxMultiRequestSize = 1;
        props->railId = -1;
        props->planeId = -1;
    }
    trace("v12.getProperties", "");
    return ncclSuccess;
}

static ncclResult_t v12_listen(void* ctx, int dev, void* handle, void** listenComm) {
    (void)ctx;(void)dev;
    sim_comm_t *lc = (sim_comm_t*)calloc(1, sizeof(*lc));
    lc->local = MY_RANK;
    lc->peer = -1;
    if (handle) {
        int *h = (int*)handle;
        h[0] = MY_RANK;
    }
    if (listenComm) *listenComm = lc;
    trace("v12.listen", "");
    return ncclSuccess;
}

static ncclResult_t v12_connect(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v12_t** sendDevComm) {
    (void)ctx;(void)dev;
    int peer = handle ? ((int*)handle)[0] : -1;
    sim_comm_t *sc = (sim_comm_t*)calloc(1, sizeof(*sc));
    sc->local = MY_RANK;
    sc->peer = peer;
    sc->chan = CHAN_SEQ++;
    if (sendComm) *sendComm = sc;
    if (sendDevComm) *sendDevComm = NULL;
    char buf[64]; snprintf(buf, sizeof(buf), "connect peer=%d", peer);
    trace("v12.connect", buf);
    return ncclSuccess;
}

static ncclResult_t v12_accept(void* listenComm, void** recvComm, ncclNetDeviceHandle_v12_t** recvDevComm) {
    sim_comm_t *lc = (sim_comm_t*)listenComm;
    sim_comm_t *rc = (sim_comm_t*)calloc(1, sizeof(*rc));
    rc->local = MY_RANK;
    /* for 2-rank, peer is the other rank */
    rc->peer = (MY_RANK == 0) ? 1 : 0;
    rc->chan = CHAN_SEQ++;
    if (recvComm) *recvComm = rc;
    if (recvDevComm) *recvDevComm = NULL;
    trace("v12.accept", "");
    return ncclSuccess;
}

static ncclResult_t v12_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
    (void)comm;(void)type;
    size_t off = fg_fab_reg_mr(FAB, data, size);
    if (off == 0) {
        /* pool full or invalid; fall back to no-op handle */
        if (mhandle) *mhandle = (void*)0x1;
    } else {
        if (mhandle) *mhandle = (void*)off;
    }
    trace("v12.regMr", "");
    return ncclSuccess;
}
static ncclResult_t v12_regMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
    (void)comm;(void)data;(void)size;(void)type;(void)offset;(void)fd;
    if (mhandle) *mhandle = (void*)0x1;
    trace("v12.regMrDmaBuf", "");
    return ncclSuccess;
}
static ncclResult_t v12_deregMr(void* comm, void* mhandle) {
    (void)comm;(void)mhandle;
    trace("v12.deregMr", "");
    return ncclSuccess;
}

static ncclResult_t v12_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
    (void)tag;(void)mhandle;(void)phandle;
    sim_comm_t *sc = (sim_comm_t*)sendComm;
    if (size > CAP) return ncclInvalidArgument;
    int slot = -1; unsigned seq = 0;
    /* Per-link timing (gpu-sim-pkb): resolve the link for this peer. */
    int link_id = -1;
    double bw_gbps = 0.0, lat_us = 0.0, jit_us = 0.0;
    int loss_ppm = 0;
    uint64_t partition_after = 0;
    double timeout_us = 0.0;
    int have_link = 0;
    const char *tp = getenv("GPUSIM_TOPOLOGY");
    if (tp && *tp) {
        int rr = plugin_resolve_link(MY_RANK, sc->peer, &link_id, &bw_gbps, &lat_us, &jit_us, &loss_ppm, &partition_after, &timeout_us);
        if (rr == -2) {
            fprintf(stderr, "plugin: zero-bandwidth link between rank %d and %d\n", MY_RANK, sc->peer);
            return ncclSystemError;
        }
        if (rr != 0) {
            fprintf(stderr, "plugin: no link between rank %d and %d\n", MY_RANK, sc->peer);
            return ncclSystemError;
        }
        have_link = 1;
    }

    /* Apply per-link delay where the bytes cross the fabric, BEFORE publishing
       the bytes, so the wire delay is actually spent (serial wire). */
    double delay_us = 0.0;
    if (have_link) {
        delay_us = (double)size / (bw_gbps * 125.0) + lat_us + jit_us;
        int rc = fg_fab_transfer(FAB, 0, size, bw_gbps, lat_us, jit_us, loss_ppm, partition_after, timeout_us, link_id);
        if (rc != FG_FAB_OK) {
            const char *reason = "unknown";
            if (rc == FG_FAB_ELOSS) reason = "loss";
            else if (rc == FG_FAB_ETIMEDOUT) reason = "timeout";
            else if (rc == FG_FAB_EPEERDEAD) reason = "peer_dead";
            fprintf(stderr, "plugin: transfer failed: %s reason=%s\n", fg_fab_errstr(rc), reason);
            /* D5: preserve the fabric error class name in the trace */
            const char *tp = getenv("GPUSIM_TRACE");
            if (tp && *tp) {
                fprintf(stderr, "fault=%s code=%s\n", reason, fg_fab_errstr(rc));
            }
            return ncclSystemError;
        }
    }

    pthread_mutex_lock(&LOCK);
    int rc;
    size_t reg_off = (size_t)(uintptr_t)mhandle;
    if (reg_off > 1) {
        /* registered buffer: send offset, skip copy */
        rc = fg_fab_send_reg(FAB, sc->peer, reg_off - 1, size, &slot, &seq);
    } else {
        rc = fg_fab_send_x(FAB, sc->peer, data, size, &slot, &seq);
    }
    pthread_mutex_unlock(&LOCK);
    if (rc != FG_FAB_OK) {
        fprintf(stderr, "isend failed: %s\n", fg_fab_errstr(rc));
        return ncclSystemError;
    }
    /* no-spec case: no link delay (memcpy speed); GPUSIM_FAB_DELAY_US is NOT
       applied, so a graph-blind timing cannot creep in (pkb item 3). */

    req_t *rq = (req_t*)calloc(1, sizeof(*rq));
    rq->done = 1;
    rq->size = size;
    rq->chan_of_req = sc->chan;
    rq->is_send = 1;
    if (request) *request = rq;
    const char *dt = getenv("GPUSIM_DTYPE"); if (!dt || !*dt) dt = "float32";
    CNT_ISEND++;
    char buf[256];
    if (have_link) {
        snprintf(buf, sizeof(buf),
                 "isend peer=%d chan=%d slot=%d seq=%u bytes=%zu digest=%016llx tag=%d dtype=%s cnt_isend=%lu acct=none link=%d delay_us=%.3f acct=per_isend",
                 sc->peer, sc->chan, slot, seq, size, fg_digest(data, size), tag, dt, CNT_ISEND, link_id, delay_us);
    } else {
        snprintf(buf, sizeof(buf),
                 "isend peer=%d chan=%d slot=%d seq=%u bytes=%zu digest=%016llx tag=%d dtype=%s cnt_isend=%lu acct=none",
                 sc->peer, sc->chan, slot, seq, size, fg_digest(data, size), tag, dt, CNT_ISEND);
    }
    trace("v12.isend", buf);
    return ncclSuccess;
}

static ncclResult_t v12_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
    (void)tags;(void)mhandles;(void)phandles;
    sim_comm_t *rcv = (sim_comm_t*)recvComm;
    if (n < 1) return ncclInvalidArgument;
    /* handle only first buffer */
    req_t *rq = (req_t*)calloc(1, sizeof(*rq));
    rq->done = 0;
    rq->size = sizes[0];
    rq->data = data[0];
    rq->peer = rcv->peer;
    rq->chan_of_req = rcv->chan;
    rq->is_send = 0;
    /* store tag in req via a field? add tag to req_t */
    if (request) *request = rq;
    const char *dt = getenv("GPUSIM_DTYPE"); if (!dt || !*dt) dt = "float32";
    /* log EVERY posted buffer so the record is complete (n can be > 1) */
    CNT_IRECV++;
    for (int bi = 0; bi < n; bi++) {
        char buf[192];
        snprintf(buf, sizeof(buf),
                 "irecv peer=%d chan=%d buf=%d/%d size=%zu tags=%d dtype=%s cnt_irecv=%lu",
                 rcv->peer, rcv->chan, bi + 1, n, sizes[bi], tags ? tags[bi] : -1, dt, CNT_IRECV);
        trace("v12.irecv", buf);
    }
    return ncclSuccess;
}

/* Non-static so the no-NCCL peer-death test (peer_death_test.c) can call the
 * PLUGIN's real mapping rather than a copy of it. */
extern "C" int ncclnetsim_map_fab_error(int rc) {
    if (rc == FG_FAB_OK) return ncclSuccess;
    if (rc == FG_FAB_EPEERDEAD) return ncclSystemError;
    if (rc == FG_FAB_ESIZE) return ncclSystemError;
    return ncclSystemError;
}

static ncclResult_t v12_test(void* request, int* done, int* sizes) {
    req_t *rq = (req_t*)request;
    if (!rq) return ncclInvalidArgument;
    if (rq->done) {
        if (done) *done = 1;
        if (sizes) *sizes = (int)rq->size;
        return ncclSuccess;
    }
    /* try receive with short timeout */
    pthread_mutex_lock(&LOCK);
    size_t got = 0;
    int slot = -1; unsigned seq = 0, consumed = 0;
    int rc = fg_fab_recv_max_ex(FAB, rq->peer, rq->data, rq->size, &got,
                                &slot, &seq, &consumed);
    pthread_mutex_unlock(&LOCK);
    if (rc == FG_FAB_OK) {
        rq->size = got;
        rq->done = 1;
        if (done) *done = 1;
        if (sizes) *sizes = (int)got;
        CNT_RECV_DONE++;
        int rlink = -1; double rbw=0, rlat=0;
        const char *tp = getenv("GPUSIM_TOPOLOGY");
        if (tp && *tp) plugin_resolve_link(rq->peer, MY_RANK, &rlink, &rbw, &rlat);
        char buf[256];
        if (rlink >= 0) {
            snprintf(buf, sizeof(buf),
                     "recv-done peer=%d chan=%d slot=%d seq=%u bytes=%zu digest=%016llx consumed_seq=%u cnt_recv_done=%lu link=%d",
                     rq->peer, rq->chan_of_req, slot, seq, got, fg_digest(rq->data, got), consumed, CNT_RECV_DONE, rlink);
        } else {
            snprintf(buf, sizeof(buf),
                     "recv-done peer=%d chan=%d slot=%d seq=%u bytes=%zu digest=%016llx consumed_seq=%u cnt_recv_done=%lu",
                     rq->peer, rq->chan_of_req, slot, seq, got, fg_digest(rq->data, got), consumed, CNT_RECV_DONE);
        }
        trace("v12.recv", buf);
        return ncclSuccess;
    }
    if (rc == FG_FAB_ESIZE) {
        fprintf(stderr, "size mismatch in recv\n");
        return (ncclResult_t)ncclnetsim_map_fab_error(rc);
    }
    if (rc == FG_FAB_EPEERDEAD) {
        return (ncclResult_t)ncclnetsim_map_fab_error(rc);
    }
    /* timeout or other: not done */
    if (done) *done = 0;
    if (sizes) *sizes = 0;
    return ncclSuccess;
}

static ncclResult_t v12_iflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
    (void)recvComm;(void)n;(void)data;(void)sizes;(void)mhandles;
    if (request) {
        req_t *rq = (req_t*)calloc(1, sizeof(*rq));
        rq->done = 1;
        *request = rq;
    }
    trace("v12.iflush", "");
    return ncclSuccess;
}

static ncclResult_t v12_closeSend(void* sendComm) { free(sendComm); trace("v12.closeSend",""); return ncclSuccess; }
static ncclResult_t v12_closeRecv(void* recvComm) { free(recvComm); trace("v12.closeRecv",""); return ncclSuccess; }
static ncclResult_t v12_closeListen(void* listenComm) { free(listenComm); trace("v12.closeListen",""); return ncclSuccess; }

static ncclResult_t v12_listen_wrap(void* ctx, int dev, void* handle, void** listenComm) { return v12_listen(ctx, dev, handle, listenComm); }
static ncclResult_t v12_connect_wrap(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v12_t** sendDevComm) { return v12_connect(ctx, dev, handle, sendComm, sendDevComm); }
static ncclResult_t v12_accept_wrap(void* listenComm, void** recvComm, ncclNetDeviceHandle_v12_t** recvDevComm) { return v12_accept(listenComm, recvComm, recvDevComm); }

/* function pointer wrappers already defined */
static ncclResult_t v12_init_wrap(void** ctx, uint64_t commId, ncclNetCommConfig_v12_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
    return v12_init(ctx, commId, config, logFunction, profFunction);
}
static ncclResult_t v12_devices_wrap(int* ndev) { return v12_devices(ndev); }
static ncclResult_t v12_getProperties_wrap(int dev, ncclNetProperties_v12_t* props) { return v12_getProperties(dev, props); }

static ncclNet_v12_t v12_plugin = {
    .name = "sim",
    .init = v12_init_wrap,
    .devices = v12_devices_wrap,
    .getProperties = v12_getProperties_wrap,
    .listen = v12_listen_wrap,
    .connect = v12_connect_wrap,
    .accept = v12_accept_wrap,
    .regMr = v12_regMr,
    .regMrDmaBuf = v12_regMrDmaBuf,
    .deregMr = v12_deregMr,
    .isend = v12_isend,
    .irecv = v12_irecv,
    .iflush = v12_iflush,
    .test = v12_test,
    .closeSend = v12_closeSend,
    .closeRecv = v12_closeRecv,
    .closeListen = v12_closeListen,
    .getDeviceMr = (ncclResult_t(*)(void*,void*,void**)) (void*)0, /* placeholder */
    .irecvConsumed = NULL,
    .makeVDevice = NULL,
    .finalize = NULL,
    .setNetAttr = NULL,
};

/* fix getDeviceMr */
static ncclResult_t v12_getDeviceMr(void* comm, void* mhandle, void** dptr) {
    (void)comm;(void)mhandle; if (dptr) *dptr = NULL; return ncclSuccess;
}
static ncclResult_t v12_finalize(void* ctx) { (void)ctx; fg_fab_detach(FAB); FAB = NULL; if (TOPO) { fg_topo_free(TOPO); TOPO = NULL; } trace("v12.finalize",""); return ncclSuccess; }

/* rebuild struct with correct pointers via constructor? Use manual assignment in init? Instead define after functions. */
/* We'll use a global initializer function instead. */

__attribute__((visibility("default"))) ncclNet_v12_t ncclNetPlugin_v12;

static ncclResult_t v12_irecvConsumed(void* recvComm, int n, void* request) {
    (void)recvComm;(void)n;(void)request; return ncclSuccess;
}
static ncclResult_t v12_makeVDevice(int* d, ncclNetVDeviceProps_v12_t* props) {
    (void)d;(void)props; return ncclSuccess;
}
static void __attribute__((constructor)) init_plugin() {
    memset(&ncclNetPlugin_v12, 0, sizeof(ncclNetPlugin_v12));
    ncclNetPlugin_v12.name = "sim";
    ncclNetPlugin_v12.init = v12_init_wrap;
    ncclNetPlugin_v12.devices = v12_devices_wrap;
    ncclNetPlugin_v12.getProperties = v12_getProperties_wrap;
    ncclNetPlugin_v12.listen = v12_listen_wrap;
    ncclNetPlugin_v12.connect = v12_connect_wrap;
    ncclNetPlugin_v12.accept = v12_accept_wrap;
    ncclNetPlugin_v12.regMr = v12_regMr;
    ncclNetPlugin_v12.regMrDmaBuf = v12_regMrDmaBuf;
    ncclNetPlugin_v12.deregMr = v12_deregMr;
    ncclNetPlugin_v12.isend = v12_isend;
    ncclNetPlugin_v12.irecv = v12_irecv;
    ncclNetPlugin_v12.iflush = v12_iflush;
    ncclNetPlugin_v12.test = v12_test;
    ncclNetPlugin_v12.closeSend = v12_closeSend;
    ncclNetPlugin_v12.closeRecv = v12_closeRecv;
    ncclNetPlugin_v12.closeListen = v12_closeListen;
    ncclNetPlugin_v12.getDeviceMr = v12_getDeviceMr;
    ncclNetPlugin_v12.irecvConsumed = v12_irecvConsumed;
    ncclNetPlugin_v12.makeVDevice = v12_makeVDevice;
    ncclNetPlugin_v12.finalize = v12_finalize;
    ncclNetPlugin_v12.setNetAttr = NULL;
}
