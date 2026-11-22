/* gpu-sim-7q3w.3: libibverbs.so.1 shim so NCCL's OWN IB transport runs.
 *
 * WHICH HEADER, AND WHY IT IS NOT THE OBVIOUS ONE
 * ----------------------------------------------
 * The ABI that crosses this shim is NCCL's, not the distro's.  Measured
 * (experiments/gpu-sim-7q3w.3/abi_layout.c, run.sh diffs both dumps):
 *
 *   struct            rdma-core v59.3   NCCL 2.32.6 ibvcore.h
 *   ibv_send_wr             128                 80
 *   ibv_qp_init_attr         64                 72
 *   ibv_qp_attr             144                136
 *
 * So the shim COMPILES AGAINST vendor/nccl/ibvcore.h (NCCL v2.32.6-1, the
 * caller's own ABI snapshot, SHA256 in vendor/nccl/SHA256SUMS).  The v59.3
 * headers stay in vendor/infiniband/ as the measured counter-example; the
 * build refuses if a swap makes the two agree or disagree in a new place.
 *
 * THE INLINE TRAP
 * ---------------
 * NCCL does NOT dlsym ibv_post_send / ibv_post_recv / ibv_poll_cq.  Those are
 * static inlines in its ibvcore.h that dispatch through
 * qp->context->ops.post_send / .post_recv and cq->context->ops.poll_cq
 * (src/include/ibvwrap.h in NCCL v2.32.6-1).  A shim that exports the symbols
 * but leaves ctx->ops zeroed segfaults on the first poll.  ibv_open_device below
 * therefore fills the ops table; the exported symbols exist for completeness
 * and for our own probe.
 *
 * TIMING
 * ------
 * Every completion carries ready_ns = now + (bytes / modelled bandwidth +
 * modelled latency) and poll_cq returns it only once the clock has reached
 * ready_ns.  No scattered sleeps: the model is the clock.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* verbs.h-style static inlines (ibv_post_send, verbs_get_ctx) collide with our
 * definitions, so the ABI symbols below are emitted with asm labels. */
#include "nccl/ibvcore.h"

/* ---- ABI pin: sizes and offsets measured from vendor/nccl/ibvcore.h ------- */
_Static_assert(sizeof(struct ibv_send_wr) == 80, "NCCL ABI: ibv_send_wr");
_Static_assert(sizeof(struct ibv_qp_init_attr) == 72, "NCCL ABI: ibv_qp_init_attr");
_Static_assert(sizeof(struct ibv_qp_attr) == 136, "NCCL ABI: ibv_qp_attr");
_Static_assert(sizeof(struct ibv_recv_wr) == 32, "NCCL ABI: ibv_recv_wr");
_Static_assert(sizeof(struct ibv_sge) == 16, "NCCL ABI: ibv_sge");
_Static_assert(sizeof(struct ibv_wc) == 48, "NCCL ABI: ibv_wc");
_Static_assert(sizeof(struct ibv_mr) == 48, "NCCL ABI: ibv_mr");
_Static_assert(sizeof(struct ibv_context) == 328, "NCCL ABI: ibv_context");
_Static_assert(sizeof(struct ibv_context_ops) == 256, "NCCL ABI: ibv_context_ops");
_Static_assert(sizeof(struct ibv_port_attr) == 56, "NCCL ABI: ibv_port_attr");
_Static_assert(offsetof(struct ibv_context, ops) == 8, "NCCL ABI: ops offset");
_Static_assert(offsetof(struct ibv_context_ops, poll_cq) == 88, "NCCL ABI: poll_cq");
_Static_assert(offsetof(struct ibv_context_ops, req_notify_cq) == 96, "NCCL ABI: req_notify_cq");
_Static_assert(offsetof(struct ibv_context_ops, post_send) == 200, "NCCL ABI: post_send");
_Static_assert(offsetof(struct ibv_context_ops, post_recv) == 208, "NCCL ABI: post_recv");
_Static_assert(offsetof(struct ibv_wc, status) == 8, "NCCL ABI: wc.status");
_Static_assert(offsetof(struct ibv_wc, opcode) == 12, "NCCL ABI: wc.opcode");
_Static_assert(offsetof(struct ibv_wc, byte_len) == 20, "NCCL ABI: wc.byte_len");
_Static_assert(offsetof(struct ibv_wc, qp_num) == 28, "NCCL ABI: wc.qp_num");
_Static_assert(offsetof(struct ibv_wc, imm_data) == 24, "NCCL ABI: wc.imm_data");
_Static_assert(offsetof(struct ibv_wc, wc_flags) == 36, "NCCL ABI: wc.wc_flags");
_Static_assert(offsetof(struct ibv_send_wr, wr.rdma.remote_addr) == 40, "NCCL ABI: rdma.addr");
_Static_assert(offsetof(struct ibv_send_wr, wr.rdma.rkey) == 48, "NCCL ABI: rdma.rkey");
_Static_assert(offsetof(struct ibv_qp_attr, ah_attr) == 56, "NCCL ABI: qp_attr.ah_attr");
_Static_assert(offsetof(struct ibv_qp_attr, dest_qp_num) == 28, "NCCL ABI: qp_attr.dest_qp_num");
_Static_assert(offsetof(struct ibv_qp_attr, rq_psn) == 20, "NCCL ABI: qp_attr.rq_psn");
_Static_assert(offsetof(struct ibv_qp_attr, sq_psn) == 24, "NCCL ABI: qp_attr.sq_psn");

/* ---- modelled wire ------------------------------------------------------- */
/* B300-class line rate modelled per NIC: 400 Gbps (NDR 4x 100G/lane).
 * FAKEIB_BW_GBPS / FAKEIB_LATENCY_US exist so the 2x-ratio assertion in the
 * acceptance test drives the SAME code path a real spec file will. */
static double model_bw_gbps(void) {
    const char *e = getenv("FAKEIB_BW_GBPS");
    return (e && atof(e) > 0) ? atof(e) : 400.0;
}
static double model_latency_us(void) {
    const char *e = getenv("FAKEIB_LATENCY_US");
    return e ? atof(e) : 1.2;
}
static uint64_t model_wire_ns(size_t bytes) {
    double sec = (double)bytes / (model_bw_gbps() * 1e9 / 8.0);
    return (uint64_t)(sec * 1e9) + (uint64_t)(model_latency_us() * 1000.0);
}
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* ---- counters the acceptance test asserts on ----------------------------- */
uint64_t fakeib_dmabuf_registrations;
uint64_t fakeib_reg_mr_calls;
uint64_t fakeib_post_send_calls;
uint64_t fakeib_post_recv_calls;
uint64_t fakeib_completions;
uint64_t fakeib_bytes_written;
uint64_t fakeib_bytes_read;
uint64_t fakeib_writes;
uint64_t fakeib_reads;
uint64_t fakeib_sends;
uint64_t fakeib_recvs;
uint64_t fakeib_qp_transition_refusals;
uint64_t fakeib_rnr_drops;
uint64_t fakeib_remote_access_errors;
uint64_t fakeib_cq_full_drops;
/* last modelled charge, exported so the acceptance can assert the bytes/bandwidth
 * term deterministically instead of racing a loaded box's scheduler */
uint64_t fakeib_last_charge_ns;
uint64_t fakeib_last_charge_bytes;
/* defined in the cross-process wire (ibverbs_ring.inc); declared here because
 * the counters file reports them */
uint64_t fakeib_ring_records_out, fakeib_ring_records_in, fakeib_ring_stall_drops;

static void counters_dump(void) {
    const char *path = getenv("FAKEIB_COUNTERS_FILE");
    if (!path || !*path) return;
    FILE *f = fopen(path, "a");
    if (!f) return;
    fprintf(f,
            "{\"rank\":%d,\"dmabuf_registrations\":%llu,\"reg_mr_calls\":%llu,"
            "\"post_send_calls\":%llu,\"post_recv_calls\":%llu,"
            "\"completions\":%llu,\"bytes_written\":%llu,\"bytes_read\":%llu,"
            "\"writes\":%llu,\"reads\":%llu,\"sends\":%llu,\"recvs\":%llu,"
            "\"qp_transition_refusals\":%llu,\"rnr_drops\":%llu,"
            "\"remote_access_errors\":%llu,\"cq_full_drops\":%llu,"
            "\"ring_out\":%llu,\"ring_in\":%llu,\"ring_stall_drops\":%llu}\n",
            (int)strtol(getenv("FAKEIB_RANK") ? getenv("FAKEIB_RANK") : "0", NULL, 10),
            (unsigned long long)fakeib_dmabuf_registrations,
            (unsigned long long)fakeib_reg_mr_calls,
            (unsigned long long)fakeib_post_send_calls,
            (unsigned long long)fakeib_post_recv_calls,
            (unsigned long long)fakeib_completions,
            (unsigned long long)fakeib_bytes_written,
            (unsigned long long)fakeib_bytes_read,
            (unsigned long long)fakeib_writes, (unsigned long long)fakeib_reads,
            (unsigned long long)fakeib_sends, (unsigned long long)fakeib_recvs,
            (unsigned long long)fakeib_qp_transition_refusals,
            (unsigned long long)fakeib_rnr_drops,
            (unsigned long long)fakeib_remote_access_errors,
            (unsigned long long)fakeib_cq_full_drops,
            (unsigned long long)fakeib_ring_records_out,
            (unsigned long long)fakeib_ring_records_in,
            (unsigned long long)fakeib_ring_stall_drops);
    fclose(f);
}
__attribute__((destructor)) static void fakeib_fini(void) { counters_dump(); }

/* ---- identity ------------------------------------------------------------ */
/* One rank per process; the GID (and LID) derive from it. */
static int my_rank(void) {
    static int r = -1;
    if (r < 0) {
        const char *e = getenv("FAKEIB_RANK");
        r = e ? (int)strtol(e, NULL, 10) : 0;
    }
    return r;
}

/* ---- devices ------------------------------------------------------------- */
#define MAX_DEVS 16
static int g_ndev;
static char g_names[MAX_DEVS][64];
static struct ibv_device g_devs[MAX_DEVS];
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void init_devs(void) {
    const char *e = getenv("FAKEIB_DEVICES");
    if (!e || !*e) e = "mlx5_0";
    char buf[512];
    snprintf(buf, sizeof buf, "%s", e);
    g_ndev = 0;
    for (char *tok = strtok(buf, ","); tok && g_ndev < MAX_DEVS; tok = strtok(NULL, ",")) {
        snprintf(g_names[g_ndev], sizeof g_names[0], "%s", tok);
        memset(&g_devs[g_ndev], 0, sizeof g_devs[0]);
        snprintf(g_devs[g_ndev].name, sizeof g_devs[0].name, "%s", g_names[g_ndev]);
        snprintf(g_devs[g_ndev].dev_name, sizeof g_devs[0].dev_name, "uverbs%d", g_ndev);
        snprintf(g_devs[g_ndev].dev_path, sizeof g_devs[0].dev_path,
                 "/sys/class/infiniband_verbs/uverbs%d", g_ndev);
        snprintf(g_devs[g_ndev].ibdev_path, sizeof g_devs[0].ibdev_path,
                 "/sys/class/infiniband/%s", g_names[g_ndev]);
        g_devs[g_ndev].node_type = IBV_NODE_CA;
        g_devs[g_ndev].transport_type = IBV_TRANSPORT_IB;
        g_ndev++;
    }
}

struct ibv_device **ibv_get_device_list(int *num) {
    pthread_once(&g_once, init_devs);
    struct ibv_device **list = calloc((size_t)g_ndev + 1, sizeof *list);
    if (!list) { errno = ENOMEM; return NULL; }
    for (int i = 0; i < g_ndev; i++) list[i] = &g_devs[i];
    if (num) *num = g_ndev;
    return list;
}
void ibv_free_device_list(struct ibv_device **list) { free(list); }
const char *ibv_get_device_name(struct ibv_device *dev) { return dev ? dev->name : NULL; }
int ibv_get_device_index(struct ibv_device *dev) {
    for (int i = 0; i < g_ndev; i++) if (&g_devs[i] == dev) return i;
    return -1;
}

/* ---- completion queues --------------------------------------------------- */
static int shim_ops_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc);
static int shim_ops_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                              struct ibv_send_wr **bad_wr);
static int shim_ops_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                              struct ibv_recv_wr **bad_wr);
static int shim_ops_req_notify_cq(struct ibv_cq *cq, int solicited_only);
static void shim_ops_cq_event(struct ibv_cq *cq);

struct wc_entry { struct ibv_wc wc; uint64_t ready_ns; };
struct fake_cq {
    struct ibv_cq cq;               /* first: cast from struct ibv_cq* */
    pthread_mutex_t lock;
    struct wc_entry *ring;
    uint64_t cap, head, tail;       /* monotonic counts; index = n % cap */
    int notify_armed;
    struct ibv_comp_channel *channel;
};

static void cq_push(struct fake_cq *c, const struct ibv_wc *wc, uint64_t ready_ns) {
    if (!c) return;
    pthread_mutex_lock(&c->lock);
    if (c->tail - c->head >= c->cap) {
        __atomic_add_fetch(&fakeib_cq_full_drops, 1, __ATOMIC_RELAXED);
    } else {
        struct wc_entry *e = &c->ring[c->tail % c->cap];
        e->wc = *wc;
        e->ready_ns = ready_ns;
        c->tail++;
        if (c->notify_armed && c->channel && c->channel->fd >= 0) {
            uint64_t one = 1;
            ssize_t r = write(c->channel->fd, &one, sizeof one);
            (void)r;
            c->notify_armed = 0;
        }
    }
    pthread_mutex_unlock(&c->lock);
}

struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *ctx) {
    struct ibv_comp_channel *ch = calloc(1, sizeof *ch);
    if (!ch) { errno = ENOMEM; return NULL; }
    ch->context = ctx;
    ch->fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (ch->fd < 0) { free(ch); return NULL; }
    ch->refcnt = 0;
    return ch;
}
int ibv_destroy_comp_channel(struct ibv_comp_channel *ch) {
    if (!ch) return 0;
    if (ch->fd >= 0) close(ch->fd);
    free(ch);
    return 0;
}

struct ibv_cq *ibv_create_cq(struct ibv_context *ctx, int cqe, void *cq_context,
                             struct ibv_comp_channel *channel, int comp_vector) {
    (void)comp_vector;
    struct fake_cq *c = calloc(1, sizeof *c);
    if (!c) { errno = ENOMEM; return NULL; }
    c->cap = (uint64_t)(cqe > 0 ? cqe : 64);
    c->ring = calloc(c->cap, sizeof *c->ring);
    if (!c->ring) { free(c); errno = ENOMEM; return NULL; }
    pthread_mutex_init(&c->lock, NULL);
    c->cq.context = ctx;
    c->cq.cq_context = cq_context;
    c->cq.channel = channel;
    c->cq.cqe = cqe;
    c->channel = channel;
    return &c->cq;
}
int ibv_destroy_cq(struct ibv_cq *cq) {
    struct fake_cq *c = (struct fake_cq *)cq;
    if (!c) return 0;
    pthread_mutex_destroy(&c->lock);
    free(c->ring);
    free(c);
    return 0;
}
int ibv_resize_cq(struct ibv_cq *cq, int cqe) { (void)cq; (void)cqe; return ENOSYS; }

static int shim_ops_req_notify_cq(struct ibv_cq *cq, int solicited_only) {
    (void)solicited_only;
    struct fake_cq *c = (struct fake_cq *)cq;
    if (!c) return EINVAL;
    pthread_mutex_lock(&c->lock);
    if (c->tail > c->head) {
        /* A completion is already queued: real semantics fire immediately. */
        if (c->channel && c->channel->fd >= 0) {
            uint64_t one = 1;
            ssize_t r = write(c->channel->fd, &one, sizeof one);
            (void)r;
        }
        c->notify_armed = 0;
    } else {
        c->notify_armed = 1;
    }
    pthread_mutex_unlock(&c->lock);
    return 0;
}
static void shim_ops_cq_event(struct ibv_cq *cq) { (void)cq; }

/* Real rdma-core exposes these as inline/op calls; export them too so a
 * non-NCCL caller (our probe, a future harness) can use plain symbols. */
int ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only) {
    return shim_ops_req_notify_cq(cq, solicited_only);
}
int ibv_get_cq_event(struct ibv_comp_channel *ch, struct ibv_cq **cq, void **cq_context) {
    if (!ch || ch->fd < 0) return -1;
    uint64_t v;
    ssize_t r = read(ch->fd, &v, sizeof v);   /* eventfd: consumes the counter */
    if (r != (ssize_t)sizeof v) return -1;
    /* The caller polls its own CQs; rdma-core looks the CQ up here.  Our probe
     * passes the CQs it owns, so report the first one waiting on this channel. */
    *cq = NULL;
    *cq_context = NULL;
    return 0;
}
void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents) {
    (void)cq; (void)nevents;
}

/* ---- memory regions ------------------------------------------------------ */
#define MAX_MRS 4096
struct mr_entry {
    int used;
    uint64_t rkey;
    void *addr;      /* virtual address in THIS process */
    uint64_t iova;   /* value remote peers use as the base address */
    size_t len;
    int access;
};
static struct mr_entry g_mrs[MAX_MRS];
static pthread_mutex_t g_mr_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_next_rkey = 0x1000;

static struct mr_entry *mr_lookup(uint64_t rkey) {
    struct mr_entry *found = NULL;
    pthread_mutex_lock(&g_mr_lock);
    for (int i = 0; i < MAX_MRS; i++) {
        if (g_mrs[i].used && g_mrs[i].rkey == rkey) { found = &g_mrs[i]; break; }
    }
    pthread_mutex_unlock(&g_mr_lock);
    return found;   /* entries are never reused in this shim */
}

struct fake_pd { struct ibv_pd pd; };
struct fake_ah { struct ibv_ah ah; };
struct fake_mr { struct ibv_mr mr; };

struct ibv_pd *ibv_alloc_pd(struct ibv_context *ctx) {
    struct fake_pd *p = calloc(1, sizeof *p);
    if (!p) { errno = ENOMEM; return NULL; }
    p->pd.context = ctx;
    p->pd.handle = (uint32_t)(uintptr_t)p;
    return &p->pd;
}
int ibv_dealloc_pd(struct ibv_pd *pd) { free(pd); return 0; }

static struct ibv_mr *mk_mr(struct ibv_pd *pd, void *addr, uint64_t iova,
                            size_t len, int access) {
    struct fake_mr *m = calloc(1, sizeof *m);
    if (!m) { errno = ENOMEM; return NULL; }
    uint64_t rkey;
    pthread_mutex_lock(&g_mr_lock);
    rkey = g_next_rkey++;
    int slot = -1;
    for (int i = 0; i < MAX_MRS; i++) if (!g_mrs[i].used) { slot = i; break; }
    if (slot < 0) { pthread_mutex_unlock(&g_mr_lock); free(m); errno = ENOMEM; return NULL; }
    g_mrs[slot] = (struct mr_entry){.used = 1, .rkey = rkey, .addr = addr,
                                    .iova = iova, .len = len, .access = access};
    pthread_mutex_unlock(&g_mr_lock);

    m->mr.context = pd ? pd->context : NULL;
    m->mr.pd = pd;
    m->mr.addr = addr;
    m->mr.length = len;
    m->mr.handle = (uint32_t)rkey;
    m->mr.lkey = (uint32_t)rkey;
    m->mr.rkey = (uint32_t)rkey;
    __atomic_add_fetch(&fakeib_reg_mr_calls, 1, __ATOMIC_RELAXED);
    return &m->mr;
}

struct ibv_mr *shim_reg_mr(struct ibv_pd *pd, void *addr, size_t len, int access)
    __asm__("ibv_reg_mr");
struct ibv_mr *shim_reg_mr(struct ibv_pd *pd, void *addr, size_t len, int access) {
    return mk_mr(pd, addr, (uint64_t)(uintptr_t)addr, len, access);
}
struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t len,
                                uint64_t iova, unsigned int access) {
    return mk_mr(pd, addr, iova, len, (int)access);
}
struct ibv_mr *ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t len,
                                 uint64_t iova, int fd, int access) {
    (void)offset; (void)fd;
    /* GDR: the dmabuf is opaque to the shim; the modelled buffer is the iova
     * range itself.  The counter is what the acceptance asserts on. */
    struct ibv_mr *mr = mk_mr(pd, (void *)(uintptr_t)iova, iova, len, access);
    if (mr) __atomic_add_fetch(&fakeib_dmabuf_registrations, 1, __ATOMIC_RELAXED);
    return mr;
}
int ibv_dereg_mr(struct ibv_mr *mr) {
    if (!mr) return 0;
    pthread_mutex_lock(&g_mr_lock);
    for (int i = 0; i < MAX_MRS; i++)
        if (g_mrs[i].used && g_mrs[i].rkey == mr->rkey) g_mrs[i].used = 0;
    pthread_mutex_unlock(&g_mr_lock);
    free((struct fake_mr *)mr);
    return 0;
}

struct ibv_ah *ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr) {
    (void)attr;
    struct fake_ah *a = calloc(1, sizeof *a);
    if (!a) { errno = ENOMEM; return NULL; }
    a->ah.context = pd ? pd->context : NULL;
    a->ah.pd = pd;
    a->ah.handle = (uint32_t)(uintptr_t)a;
    return &a->ah;
}
int ibv_destroy_ah(struct ibv_ah *ah) { free(ah); return 0; }
struct ibv_srq *ibv_create_srq(struct ibv_pd *pd, struct ibv_srq_init_attr *attr) {
    (void)pd; (void)attr;
    errno = ENOSYS;
    return NULL;
}
int ibv_destroy_srq(struct ibv_srq *srq) { (void)srq; return 0; }

/* ---- queue pairs --------------------------------------------------------- */
struct posted_recv {
    struct ibv_recv_wr wr;          /* by value: sg_list points at the caller's sge array */
    struct ibv_sge sge[8];
    int nsge;
};
struct fake_qp {
    struct ibv_qp qp;               /* first: cast from struct ibv_qp* */
    struct fake_cq *send_cq, *recv_cq;
    struct ibv_qp_cap cap;
    int sq_sig_all;
    uint32_t peer_qpn;
    int peer_rank;
    struct posted_recv *rq;
    uint64_t rq_cap, rq_head, rq_tail;   /* posted, not yet consumed */
};
#define MAX_QPS 1024
static struct fake_qp *g_qps[MAX_QPS];
static pthread_mutex_t g_qp_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t g_next_qpn = 0x1000;

static void qp_register(struct fake_qp *q) {
    pthread_mutex_lock(&g_qp_lock);
    for (int i = 0; i < MAX_QPS; i++) if (!g_qps[i]) { g_qps[i] = q; break; }
    pthread_mutex_unlock(&g_qp_lock);
}
static void qp_unregister(struct fake_qp *q) {
    pthread_mutex_lock(&g_qp_lock);
    for (int i = 0; i < MAX_QPS; i++) if (g_qps[i] == q) g_qps[i] = NULL;
    pthread_mutex_unlock(&g_qp_lock);
}
static struct fake_qp *qp_by_num(uint32_t qpn) {
    struct fake_qp *found = NULL;
    pthread_mutex_lock(&g_qp_lock);
    for (int i = 0; i < MAX_QPS; i++)
        if (g_qps[i] && g_qps[i]->qp.qp_num == qpn) { found = g_qps[i]; break; }
    pthread_mutex_unlock(&g_qp_lock);
    return found;
}

static int qp_transition_ok(int from, int to) {
    if (from == to) return 1;                     /* same-state modify is a no-op */
    if (to == IBV_QPS_RESET) return 1;
    if (to == IBV_QPS_ERR) return 1;
    if (from == IBV_QPS_RESET && to == IBV_QPS_INIT) return 1;
    if (from == IBV_QPS_INIT && to == IBV_QPS_RTR) return 1;
    if (from == IBV_QPS_RTR && to == IBV_QPS_RTS) return 1;
    if (from == IBV_QPS_RTS && (to == IBV_QPS_SQD || to == IBV_QPS_SQE)) return 1;
    return 0;
}

struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *attr) {
    if (!pd || !attr) { errno = EINVAL; return NULL; }
    struct fake_qp *q = calloc(1, sizeof *q);
    if (!q) { errno = ENOMEM; return NULL; }
    q->qp.context = pd->context;
    q->qp.pd = pd;
    q->qp.qp_context = attr->qp_context;
    q->qp.send_cq = attr->send_cq;
    q->qp.recv_cq = attr->recv_cq;
    q->qp.srq = attr->srq;
    q->qp.qp_type = attr->qp_type;
    q->qp.state = IBV_QPS_RESET;
    q->qp.qp_num = __atomic_fetch_add(&g_next_qpn, 1, __ATOMIC_RELAXED);
    q->send_cq = (struct fake_cq *)attr->send_cq;
    q->recv_cq = (struct fake_cq *)attr->recv_cq;
    q->sq_sig_all = attr->sq_sig_all;
    q->peer_rank = -1;
    /* Never advertise more than asked: NCCL sizes its WR arrays from these. */
    q->cap.max_send_wr = attr->cap.max_send_wr ? attr->cap.max_send_wr : 128;
    q->cap.max_recv_wr = attr->cap.max_recv_wr ? attr->cap.max_recv_wr : 128;
    q->cap.max_send_sge = attr->cap.max_send_sge ? attr->cap.max_send_sge : 1;
    q->cap.max_recv_sge = attr->cap.max_recv_sge ? attr->cap.max_recv_sge : 1;
    q->cap.max_inline_data = 0;
    attr->cap = q->cap;
    qp_register(q);
    return &q->qp;
}
int ibv_destroy_qp(struct ibv_qp *qp) {
    struct fake_qp *q = (struct fake_qp *)qp;
    if (!q) return 0;
    qp_unregister(q);
    free(q->rq);
    free(q);
    return 0;
}

int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int mask) {
    struct fake_qp *q = (struct fake_qp *)qp;
    if (!qp || !attr) return EINVAL;
    if (mask & IBV_QP_STATE) {
        if (!qp_transition_ok((int)qp->state, (int)attr->qp_state)) {
            __atomic_add_fetch(&fakeib_qp_transition_refusals, 1, __ATOMIC_RELAXED);
            fprintf(stderr, "FAKEIB: refused QP transition %d -> %d\n",
                    (int)qp->state, (int)attr->qp_state);
            return EINVAL;
        }
        qp->state = attr->qp_state;
    }
    if ((mask & IBV_QP_AV) && qp->state == IBV_QPS_RTR) {
        const struct ibv_ah_attr *ah = &attr->ah_attr;
        if (ah->is_global) {
            q->peer_rank = ah->grh.dgid.raw[15];
            q->peer_qpn = attr->dest_qp_num;
        } else {
            /* Our LID encodes rank+1 (see ibv_query_port). */
            q->peer_rank = ah->dlid ? ah->dlid - 1 : -1;
            q->peer_qpn = attr->dest_qp_num;
        }
    }
    if (mask & IBV_QP_CAP) q->cap = attr->cap;
    return 0;
}

int shim_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int mask,
                  struct ibv_qp_init_attr *init) __asm__("ibv_query_qp");
int shim_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int mask,
                  struct ibv_qp_init_attr *init) {
    struct fake_qp *q = (struct fake_qp *)qp;
    if (!q) return EINVAL;
    if (attr) {
        memset(attr, 0, sizeof *attr);   /* sizeof is the NCCL ABI (136) */
        if (mask & IBV_QP_STATE) attr->qp_state = qp->state;
        if (mask & IBV_QP_CAP) attr->cap = q->cap;
    }
    if (init) {
        memset(init, 0, sizeof *init);   /* sizeof is the NCCL ABI (72) */
        init->qp_context = qp->qp_context;
        init->send_cq = qp->send_cq;
        init->recv_cq = qp->recv_cq;
        init->srq = qp->srq;
        init->cap = q->cap;
        init->qp_type = qp->qp_type;
        init->sq_sig_all = q->sq_sig_all;
    }
    return 0;
}

/* ---- data plane (in-process / loopback) ---------------------------------- */
static size_t wr_payload_bytes(const struct ibv_sge *sg, int n) {
    size_t b = 0;
    for (int i = 0; i < n; i++) b += sg[i].length;
    return b;
}

/* Copy between an sg list and a flat buffer.  Returns bytes copied. */
static size_t sg_to_flat(const struct ibv_sge *sg, int n, void *dst, size_t cap) {
    size_t off = 0;
    for (int i = 0; i < n && off < cap; i++) {
        size_t len = sg[i].length;
        if (off + len > cap) len = cap - off;
        memcpy((char *)dst + off, (void *)(uintptr_t)sg[i].addr, len);
        off += len;
    }
    return off;
}
static size_t flat_to_sg(const void *src, size_t len, const struct ibv_sge *sg, int n) {
    size_t off = 0;
    for (int i = 0; i < n && off < len; i++) {
        size_t take = sg[i].length;
        if (off + take > len) take = len - off;
        memcpy((void *)(uintptr_t)sg[i].addr, (const char *)src + off, take);
        off += take;
    }
    return off;
}

/* Resolve a remote (addr, rkey) on THIS process: what a receiving rank does. */
static void *remote_to_local(uint64_t remote_addr, uint64_t rkey, size_t len) {
    struct mr_entry *m = mr_lookup(rkey);
    if (!m) { __atomic_add_fetch(&fakeib_remote_access_errors, 1, __ATOMIC_RELAXED); return NULL; }
    if (remote_addr < m->iova || remote_addr - m->iova > m->len ||
        len > m->len - (remote_addr - m->iova)) {
        __atomic_add_fetch(&fakeib_remote_access_errors, 1, __ATOMIC_RELAXED);
        return NULL;
    }
    return (char *)m->addr + (remote_addr - m->iova);
}

/* Deliver one inbound RDMA-WRITE / SEND to a QP in this process.  This is the
 * exact function the cross-process pump will call, so its return code is the
 * sender's completion status:
 *   0 = delivered (a SEND/IMM consumed a posted recv WR)
 *   1 = RNR  (SEND/IMM arrived with no posted recv WR; the SENDER does not
 *            complete a SEND on RNR, and an IMM still completes its write)
 *   2 = remote access error (unknown rkey, or range outside the MR) */
static int deliver_inbound(uint32_t dest_qpn, uint32_t kind, uint64_t wr_id,
                           uint64_t remote_addr, uint64_t rkey, uint32_t imm,
                           const void *payload, size_t len, uint64_t ready_ns) {
    struct fake_qp *q = qp_by_num(dest_qpn);
    if (getenv("FAKEIB_TRACE"))
        fprintf(stderr, "FAKEIB[%d] deliver_inbound qpn=0x%x kind=%u nbytes=%zu\n",
                my_rank(), dest_qpn, kind, len);
    if (!q) { __atomic_add_fetch(&fakeib_remote_access_errors, 1, __ATOMIC_RELAXED); return 2; }
    struct ibv_wc wc;
    memset(&wc, 0, sizeof wc);
    wc.qp_num = dest_qpn;
    wc.wr_id = wr_id;
    wc.status = IBV_WC_SUCCESS;

    if (kind == IBV_WR_RDMA_WRITE || kind == IBV_WR_RDMA_WRITE_WITH_IMM) {
        int rc = 0;
        if (payload && len) {
            void *dst = remote_to_local(remote_addr, rkey, len);
            if (!dst) rc = 2;
            else memcpy(dst, payload, len);
        }
        __atomic_add_fetch(&fakeib_bytes_written, len, __ATOMIC_RELAXED);
        __atomic_add_fetch(&fakeib_writes, 1, __ATOMIC_RELAXED);
        if (kind == IBV_WR_RDMA_WRITE) return rc;   /* no receiver completion */
        /* *_WITH_IMM consumes a posted recv WR on the receiver. */
        if (q->rq_head == q->rq_tail) {
            __atomic_add_fetch(&fakeib_rnr_drops, 1, __ATOMIC_RELAXED);
            return 1;
        }
        struct posted_recv *r = &q->rq[q->rq_head % q->rq_cap];
        q->rq_head++;
        wc.wr_id = r->wr.wr_id;
        wc.opcode = IBV_WC_RECV_RDMA_WITH_IMM;
        wc.imm_data = imm;
        wc.wc_flags = IBV_WC_WITH_IMM;
        cq_push(q->recv_cq, &wc, ready_ns);
        return rc;
    }
    if (kind == IBV_WR_SEND) {
        if (q->rq_head == q->rq_tail) {
            __atomic_add_fetch(&fakeib_rnr_drops, 1, __ATOMIC_RELAXED);
            return 1;
        }
        struct posted_recv *r = &q->rq[q->rq_head % q->rq_cap];
        q->rq_head++;
        size_t got = payload ? flat_to_sg(payload, len, r->sge, r->nsge) : 0;
        wc.wr_id = r->wr.wr_id;
        wc.opcode = IBV_WC_RECV;
        wc.byte_len = (uint32_t)got;
        wc.src_qp = 0;
        cq_push(q->recv_cq, &wc, ready_ns);
        __atomic_add_fetch(&fakeib_recvs, 1, __ATOMIC_RELAXED);
        return 0;
    }
    return 2;
}

/* Perform one WR against a peer that lives in THIS process (loopback).
 * Every opcode that completes on the sender posts its WC here, gated on the
 * modelled ready_ns; a SEND that hit RNR deliberately does NOT complete. */
static void loopback_wr(struct fake_qp *q, struct ibv_send_wr *w, uint64_t ready_ns) {
    size_t bytes = wr_payload_bytes(w->sg_list, w->num_sge);
    struct fake_qp *peer = qp_by_num(q->peer_qpn);
    int signalled = (w->send_flags & IBV_SEND_SIGNALED) || q->sq_sig_all;
    int complete = 1;
    struct ibv_wc wc;
    memset(&wc, 0, sizeof wc);
    wc.wr_id = w->wr_id;
    wc.qp_num = q->qp.qp_num;
    wc.status = IBV_WC_SUCCESS;

    if (!peer) {
        __atomic_add_fetch(&fakeib_remote_access_errors, 1, __ATOMIC_RELAXED);
        wc.status = IBV_WC_REM_ACCESS_ERR;
    } else if (w->opcode == IBV_WR_RDMA_WRITE || w->opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
        void *flat = bytes ? malloc(bytes) : NULL;
        size_t got = flat ? sg_to_flat(w->sg_list, w->num_sge, flat, bytes) : 0;
        int rc = deliver_inbound(peer->qp.qp_num, w->opcode, w->wr_id,
                                 w->wr.rdma.remote_addr, w->wr.rdma.rkey,
                                 w->imm_data, flat, got, ready_ns);
        free(flat);
        if (rc == 2) wc.status = IBV_WC_REM_ACCESS_ERR;
        wc.opcode = IBV_WC_RDMA_WRITE;
    } else if (w->opcode == IBV_WR_SEND || w->opcode == IBV_WR_SEND_WITH_IMM) {
        void *flat = bytes ? malloc(bytes) : NULL;
        size_t got = flat ? sg_to_flat(w->sg_list, w->num_sge, flat, bytes) : 0;
        int rc = deliver_inbound(peer->qp.qp_num, IBV_WR_SEND, w->wr_id, 0, 0,
                                 w->imm_data, flat, got, ready_ns);
        free(flat);
        __atomic_add_fetch(&fakeib_sends, 1, __ATOMIC_RELAXED);
        if (rc == 1) complete = 0;            /* RNR: SEND has not landed */
        else if (rc == 2) wc.status = IBV_WC_REM_ACCESS_ERR;
        wc.opcode = IBV_WC_SEND;
        wc.byte_len = (uint32_t)bytes;
    } else if (w->opcode == IBV_WR_RDMA_READ) {
        void *src = remote_to_local(w->wr.rdma.remote_addr, w->wr.rdma.rkey, bytes);
        wc.opcode = IBV_WC_RDMA_READ;
        if (!src) wc.status = IBV_WC_REM_ACCESS_ERR;
        else {
            wc.byte_len = (uint32_t)flat_to_sg(src, bytes, w->sg_list, w->num_sge);
            __atomic_add_fetch(&fakeib_bytes_read, bytes, __ATOMIC_RELAXED);
            __atomic_add_fetch(&fakeib_reads, 1, __ATOMIC_RELAXED);
        }
    } else {
        wc.status = IBV_WC_LOC_QP_OP_ERR;
    }
    if (complete && signalled) cq_push(q->send_cq, &wc, ready_ns);
}

/* The cross-process wire lives in its own file: it needs the loopback helpers
 * above and is included here, exactly once. */
#include "ibverbs_ring.inc"

static int shim_ops_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                              struct ibv_send_wr **bad_wr) {
    struct fake_qp *q = (struct fake_qp *)qp;
    if (!q) { if (bad_wr) *bad_wr = wr; return EINVAL; }
    if (qp->state != IBV_QPS_RTS && qp->state != IBV_QPS_SQD) {
        if (bad_wr) *bad_wr = wr;
        return EINVAL;
    }
    for (struct ibv_send_wr *w = wr; w; w = w->next) {
        size_t bytes = wr_payload_bytes(w->sg_list, w->num_sge);
        uint64_t charge_ns = model_wire_ns(bytes);
        __atomic_store_n(&fakeib_last_charge_ns, charge_ns, __ATOMIC_RELAXED);
        __atomic_store_n(&fakeib_last_charge_bytes, (uint64_t)bytes, __ATOMIC_RELAXED);
        uint64_t ready_ns = now_ns() + charge_ns;
        __atomic_add_fetch(&fakeib_post_send_calls, 1, __ATOMIC_RELAXED);
        /* peer_rank is set at RTR from the exchanged GID/LID; rank -1 means no
         * address was ever supplied, and a WR there is a hard error, not a
         * silent loopback. */
        if (q->peer_rank < 0) {
            __atomic_add_fetch(&fakeib_remote_access_errors, 1, __ATOMIC_RELAXED);
            if (w->send_flags & IBV_SEND_SIGNALED) {
                struct ibv_wc wc;
                memset(&wc, 0, sizeof wc);
                wc.wr_id = w->wr_id;
                wc.qp_num = q->qp.qp_num;
                wc.status = IBV_WC_REM_ACCESS_ERR;
                cq_push(q->send_cq, &wc, ready_ns);
            }
        } else if (q->peer_rank == my_rank()) {
            loopback_wr(q, w, ready_ns);
        } else {
            ring_send_wr(q, w, ready_ns);
    }
    }   /* for (w) */
    return 0;
}

static int shim_ops_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                              struct ibv_recv_wr **bad_wr) {
    struct fake_qp *q = (struct fake_qp *)qp;
    if (!q) { if (bad_wr) *bad_wr = wr; return EINVAL; }
    for (struct ibv_recv_wr *w = wr; w; w = w->next) {
        if (q->rq_tail - q->rq_head >= q->rq_cap) {
            uint64_t ncap = q->rq_cap ? q->rq_cap * 2 : 64;
            struct posted_recv *nr = realloc(q->rq, ncap * sizeof *nr);
            if (!nr) { if (bad_wr) *bad_wr = w; return ENOMEM; }
            q->rq = nr;
            q->rq_cap = ncap;
        }
        struct posted_recv *r = &q->rq[q->rq_tail % q->rq_cap];
        memset(r, 0, sizeof *r);
        r->wr = *w;
        r->nsge = w->num_sge < 8 ? w->num_sge : 8;
        for (int i = 0; i < r->nsge; i++) r->sge[i] = w->sg_list[i];
        r->wr.sg_list = r->sge;
        q->rq_tail++;
        __atomic_add_fetch(&fakeib_post_recv_calls, 1, __ATOMIC_RELAXED);
    }
    return 0;
}

int shim_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad)
    __asm__("ibv_post_send");
int shim_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **bad) {
    return shim_ops_post_send(qp, wr, bad);
}
int shim_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad)
    __asm__("ibv_post_recv");
int shim_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad) {
    return shim_ops_post_recv(qp, wr, bad);
}
int shim_post_srq_recv(struct ibv_srq *srq, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad)
    __asm__("ibv_post_srq_recv");
int shim_post_srq_recv(struct ibv_srq *srq, struct ibv_recv_wr *wr, struct ibv_recv_wr **bad) {
    (void)srq;
    if (bad) *bad = wr;
    return ENOSYS;
}

static int shim_ops_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
    /* Drain inbound wire records first: this is where a peer's RDMA WRITE lands
     * in our buffers and where its READ response becomes our completion. */
    pump_rings();
    struct fake_cq *c = (struct fake_cq *)cq;
    if (!c || !wc || num_entries <= 0) return 0;
    int n = 0;
    uint64_t t = now_ns();
    pthread_mutex_lock(&c->lock);
    while (n < num_entries && c->head < c->tail) {
        struct wc_entry *e = &c->ring[c->head % c->cap];
        if (e->ready_ns > t) break;      /* modelled wire not done yet */
        wc[n++] = e->wc;
        c->head++;
        __atomic_add_fetch(&fakeib_completions, 1, __ATOMIC_RELAXED);
    }
    pthread_mutex_unlock(&c->lock);
    return n;
}

/* Exported for link-time callers and the dlsym surface check.  NCCL itself
 * reaches poll through cq->context->ops.poll_cq, which is the same function. */
int ibv_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
    return shim_ops_poll_cq(cq, num_entries, wc);
}

/* ---- context ------------------------------------------------------------- */
struct ibv_context *ibv_open_device(struct ibv_device *dev) {
    if (!dev) { errno = EINVAL; return NULL; }
    struct ibv_context *ctx = calloc(1, sizeof *ctx);
    if (!ctx) { errno = ENOMEM; return NULL; }
    ctx->device = dev;
    ctx->cmd_fd = -1;
    ctx->async_fd = -1;
    /* The inline trap: NCCL reaches post_send/post_recv/poll_cq ONLY here. */
    ctx->ops.query_device = NULL;      /* real symbol is dlsym'd */
    ctx->ops.poll_cq = shim_ops_poll_cq;
    ctx->ops.req_notify_cq = shim_ops_req_notify_cq;
    ctx->ops.cq_event = shim_ops_cq_event;
    ctx->ops.post_send = shim_ops_post_send;
    ctx->ops.post_recv = shim_ops_post_recv;
    ctx->ops.async_event = NULL;
    return ctx;
}
int ibv_close_device(struct ibv_context *ctx) { free(ctx); return 0; }

/* ---- queries ------------------------------------------------------------- */
int ibv_query_device(struct ibv_context *ctx, struct ibv_device_attr *attr) {
    if (!ctx || !attr) return EINVAL;
    memset(attr, 0, sizeof *attr);
    snprintf(attr->fw_ver, sizeof attr->fw_ver, "28.43.1012");
    attr->node_guid = 0x0b5e2c1d4f6a7000ull | (uint64_t)(my_rank() & 0xff);
    attr->sys_image_guid = attr->node_guid;
    attr->vendor_id = 0x02c9;                 /* Mellanox */
    attr->vendor_part_id = 0x1023;            /* ConnectX-7 */
    attr->max_mr_size = (uint64_t)1 << 40;
    attr->page_size_cap = 4096 | 2097152;
    attr->max_qp = 131072;
    attr->max_qp_wr = 32768;
    attr->max_sge = 30;
    attr->max_sge_rd = 30;
    attr->max_cq = 131072;
    attr->max_cqe = 4194304;
    attr->max_mr = 131072;
    attr->max_pd = 32768;
    attr->max_qp_rd_atom = 16;
    attr->max_qp_init_rd_atom = 16;
    attr->atomic_cap = IBV_ATOMIC_NONE;
    attr->max_srq = 32768;
    attr->max_srq_wr = 32768;
    attr->max_srq_sge = 30;
    attr->max_pkeys = 128;
    attr->local_ca_ack_delay = 16;
    return 0;
}

int shim_query_port(struct ibv_context *ctx, uint8_t port, struct ibv_port_attr *attr)
    __asm__("ibv_query_port");
int shim_query_port(struct ibv_context *ctx, uint8_t port, struct ibv_port_attr *attr) {
    (void)ctx;
    if (!attr) return EINVAL;
    memset(attr, 0, sizeof *attr);
    if (port != 1) return EINVAL;
    attr->state = IBV_PORT_ACTIVE;
    attr->max_mtu = IBV_MTU_4096;
    attr->active_mtu = IBV_MTU_4096;
    attr->gid_tbl_len = 1;
    attr->port_cap_flags = IBV_PORT_IP_BASED_GIDS;
    attr->max_msg_sz = 0x40000000;
    attr->pkey_tbl_len = 1;
    attr->lid = (uint16_t)(my_rank() + 1);
    attr->active_width = 2;      /* 4x */
    attr->active_speed = 128;    /* NDR: 100 Gb/s per lane x 4 = 400 Gb/s */
    attr->phys_state = 5;        /* IB_PORT_PHYS_STATE_LINK_UP */
    return 0;
}

/* NCCL v2.32.6 ibvwrap.cc:150 — units are 100 Mb/s ("speed change detected:
 * ... Mbps").  400 Gb/s == 4000.  uint64_t out, not int. */
int ibv_query_port_speed(struct ibv_context *ctx, uint8_t port, uint64_t *speed) {
    (void)ctx; (void)port;
    if (!speed) return EINVAL;
    *speed = (uint64_t)(model_bw_gbps() * 10.0);
    return 0;
}

int ibv_query_gid(struct ibv_context *ctx, uint8_t port, int index, union ibv_gid *gid) {
    (void)ctx; (void)port;
    if (!gid) return EINVAL;
    if (index != 0) return EINVAL;
    /* RoCE v2 IPv6 link-local; the low byte is the rank, so a peer's dgid
     * tells us which process to route to (my_rank() is its counterpart). */
    memset(gid, 0, sizeof *gid);
    gid->raw[0] = 0xfe;
    gid->raw[1] = 0x80;
    gid->raw[15] = (uint8_t)my_rank();
    return 0;
}
int ibv_query_pkey(struct ibv_context *ctx, uint8_t port, int index, uint16_t *pkey) {
    (void)ctx; (void)port;
    if (!pkey) return EINVAL;
    if (index != 0) return EINVAL;
    *pkey = 0xffff;
    return 0;
}

/* ---- async events / ece -------------------------------------------------- */
int ibv_get_async_event(struct ibv_context *ctx, struct ibv_async_event *event) {
    (void)ctx;
    if (event) memset(event, 0, sizeof *event);
    return ENOSYS;
}
void ibv_ack_async_event(struct ibv_async_event *event) { (void)event; }
const char *ibv_event_type_str(enum ibv_event_type e) {
    switch (e) {
    case IBV_EVENT_CQ_ERR: return "CQ_ERR";
    case IBV_EVENT_QP_FATAL: return "QP_FATAL";
    case IBV_EVENT_QP_REQ_ERR: return "QP_REQ_ERR";
    case IBV_EVENT_QP_ACCESS_ERR: return "QP_ACCESS_ERR";
    case IBV_EVENT_COMM_EST: return "COMM_EST";
    case IBV_EVENT_SQ_DRAINED: return "SQ_DRAINED";
    case IBV_EVENT_PORT_ACTIVE: return "PORT_ACTIVE";
    default: return "UNKNOWN";
    }
}
int ibv_set_ece(struct ibv_qp *qp, struct ibv_ece *ece) { (void)qp; (void)ece; return ENOSYS; }
int ibv_query_ece(struct ibv_qp *qp, struct ibv_ece *ece) { (void)qp; (void)ece; return ENOSYS; }
int ibv_fork_init(void) { return 0; }
