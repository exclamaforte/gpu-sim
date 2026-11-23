/* gpu-sim-7q3w.3: the cross-process wire, included by ibverbs_shim.c after the
 * loopback data plane (it uses wr_payload_bytes/sg_to_flat/flat_to_sg,
 * deliver_inbound and cq_push, all defined above the include point).
 *
 * One MAP_SHARED byte ring per (from,to) rank pair, created lazily by whichever
 * side touches it first (O_EXCL decides; the opener waits for the creator's
 * magic).  A record carries the REMOTE-side coordinates exactly as NCCL
 * exchanged them (dest qpn, remote_addr, rkey); the receiving rank resolves
 * rkey in its OWN MR table, which is why two processes may both have rkey 1.
 *
 * Flow control: a full ring makes post_send spin for up to ~30 s and then
 * fail the WR with IBV_WC_LOC_QP_OP_ERR (counted), never silently drop.  The
 * ring is 64 MiB by default (FAKEIB_RING_MB) and lives in FAKEIB_RING_DIR
 * (project-local scratch), so it survives separate mount namespaces.
 *
 * The modelled delay is NOT applied to the bytes: they are in the ring at post
 * time.  It is applied to COMPLETION readiness (rec.ready_ns is absolute
 * CLOCK_MONOTONIC, and both taps share the same clock).  That is what makes the
 * collective take modelled time without the shim sleeping in a hot path.
 */
#define RING_MAGIC 0x4641494252494e47ull   /* "FAIBRING" */
#define RING_MAX 128
#define RING_TAIL_RESERVE 256u
enum ring_kind {
    RING_KIND_PAD = 1, RING_KIND_WRITE, RING_KIND_WRITE_IMM, RING_KIND_SEND,
    RING_KIND_READ_REQ, RING_KIND_READ_RESP,
};
struct ring_hdr {
    uint64_t magic;
    uint64_t cap;                  /* data bytes after this header */
    uint64_t head, tail;           /* byte counters, guarded by h->lock */
    pthread_mutex_t lock;          /* PTHREAD_PROCESS_SHARED */
};
struct ring_rec {
    uint32_t len;                  /* total record bytes, 8-aligned */
    uint32_t kind;
    uint64_t wr_id;
    uint64_t addr;
    uint64_t rkey;
    uint32_t imm;
    uint32_t dest_qpn;
    uint64_t ready_ns;
    uint32_t nbytes;
    uint32_t status;
    uint64_t tag;
};
struct ring { int fd; struct ring_hdr *h; char *data; uint64_t cap; int from, to; };
static struct ring *g_rings[RING_MAX];
static int g_nrings;
static pthread_mutex_t g_rings_lock = PTHREAD_MUTEX_INITIALIZER;
/* the counters are declared in ibverbs_shim.c (counters_dump reports them) */

static uint64_t ring_cap_bytes(void) {
    const char *e = getenv("FAKEIB_RING_MB");
    long mb = (e && atol(e) > 0) ? atol(e) : 64;
    return (uint64_t)mb << 20;
}
static void ring_mkdirs(const char *path) {
    char tmp[512];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0; mkdir(tmp, 0777); *p = '/';
    }
    mkdir(tmp, 0777);
}
static struct ring *ring_open(int from, int to) {
    pthread_mutex_lock(&g_rings_lock);
    for (int i = 0; i < g_nrings; i++)
        if (g_rings[i]->from == from && g_rings[i]->to == to) {
            struct ring *r = g_rings[i];
            pthread_mutex_unlock(&g_rings_lock);
            return r;
        }
    pthread_mutex_unlock(&g_rings_lock);

    const char *dir = getenv("FAKEIB_RING_DIR");
    if (!dir || !*dir) dir = ".lab/tmp/7q3w3-rings";
    ring_mkdirs(dir);
    char path[600];
    snprintf(path, sizeof path, "%s/ring-%d-%d.bin", dir, from, to);
    uint64_t cap = ring_cap_bytes();
    size_t total = sizeof(struct ring_hdr) + cap;

    int creator = 0;
    int fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0666);
    if (fd >= 0) creator = 1;
    else if (errno == EEXIST) fd = open(path, O_RDWR, 0666);
    if (fd < 0) return NULL;
    struct stat st;
    memset(&st, 0, sizeof st);
    if (creator) {
        if (ftruncate(fd, (off_t)total) != 0) { close(fd); return NULL; }
        if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    } else {
        for (int i = 0; i < 3000; i++) {     /* creator may still be ftruncating */
            if (fstat(fd, &st) == 0 && (size_t)st.st_size >= total) break;
            struct timespec ts = {0, 1000000};
            nanosleep(&ts, NULL);
        }
        if ((size_t)st.st_size < total) { close(fd); return NULL; }
    }
    void *m = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { close(fd); return NULL; }
    struct ring_hdr *h = m;
    if (creator) {
        pthread_mutexattr_t a;
        pthread_mutexattr_init(&a);
        pthread_mutexattr_setpshared(&a, PTHREAD_PROCESS_SHARED);
        pthread_mutex_init(&h->lock, &a);
        pthread_mutexattr_destroy(&a);
        h->cap = cap;
        h->head = h->tail = 0;
        __atomic_store_n(&h->magic, RING_MAGIC, __ATOMIC_RELEASE);
    } else {
        for (int i = 0; i < 3000; i++) {
            if (__atomic_load_n(&h->magic, __ATOMIC_ACQUIRE) == RING_MAGIC) break;
            struct timespec ts = {0, 1000000};
            nanosleep(&ts, NULL);
        }
        if (__atomic_load_n(&h->magic, __ATOMIC_ACQUIRE) != RING_MAGIC || h->cap != cap) {
            munmap(m, total);
            close(fd);
            return NULL;
        }
    }
    struct ring *r = calloc(1, sizeof *r);
    if (!r) { munmap(m, total); close(fd); return NULL; }
    r->fd = fd; r->h = h; r->data = (char *)m + sizeof(struct ring_hdr);
    r->cap = cap; r->from = from; r->to = to;
    if (getenv("FAKEIB_TRACE"))
        fprintf(stderr, "FAKEIB[%d] ring_open(%d,%d) path=%s creator=%d cap=%llu\n",
                my_rank(), from, to, path, creator, (unsigned long long)cap);
    pthread_mutex_lock(&g_rings_lock);
    if (g_nrings < RING_MAX) g_rings[g_nrings++] = r;
    pthread_mutex_unlock(&g_rings_lock);
    return r;
}

static uint64_t ring_data_cap(const struct ring *r) { return r->cap - RING_TAIL_RESERVE; }

static int ring_put(struct ring *r, const struct ring_rec *rec, const void *payload) {
    if (!r) return EAGAIN;
    uint64_t total = ((uint64_t)sizeof *rec + rec->nbytes + 7ull) & ~7ull;
    uint64_t dcap = ring_data_cap(r);
    if (total > dcap / 2) return EINVAL;      /* one record must leave room to drain */
    pthread_mutex_lock(&r->h->lock);
    for (int spins = 0;; spins++) {
        uint64_t used = r->h->tail - r->h->head;
        uint64_t off = r->h->tail % r->cap;
        uint64_t pad = (off + total > dcap) ? dcap - off : 0;
        if (r->h->cap - used >= total + pad) {
            if (pad) {
                struct ring_rec z;
                memset(&z, 0, sizeof z);
                z.len = (uint32_t)pad;
                z.kind = RING_KIND_PAD;
                memcpy(r->data + off, &z, sizeof z);
                r->h->tail += pad;
                off = 0;
            }
            /* len is the RING's framing field, not the caller's: the reader
             * advances head by it, so a zero here re-delivers the record
             * forever (measured: the receiver wedged in pump_rings). */
            struct ring_rec hdr = *rec;
            hdr.len = (uint32_t)total;
            memcpy(r->data + off, &hdr, sizeof hdr);
            if (rec->nbytes && payload)
                memcpy(r->data + off + sizeof *rec, payload, rec->nbytes);
            r->h->tail += total;
            pthread_mutex_unlock(&r->h->lock);
            __atomic_add_fetch(&fakeib_ring_records_out, 1, __ATOMIC_RELAXED);
            if (getenv("FAKEIB_TRACE"))
                fprintf(stderr, "FAKEIB[%d] ring_put(%d,%d) kind=%u nbytes=%u tail=%llu\n",
                        my_rank(), r->from, r->to, rec->kind, rec->nbytes,
                        (unsigned long long)r->h->tail);
            return 0;
        }
        pthread_mutex_unlock(&r->h->lock);
        if (spins >= 30000) {                /* ~30 s: peer is not draining */
            __atomic_add_fetch(&fakeib_ring_stall_drops, 1, __ATOMIC_RELAXED);
            return EAGAIN;
        }
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, NULL);
        pthread_mutex_lock(&r->h->lock);
    }
}

/* Non-destructive header copy; payload points INTO the mapping and stays valid
 * until the next ring_get on this ring, so the caller must consume it first. */
static int ring_get(struct ring *r, struct ring_rec *rec, const char **payload) {
    int got = 0;
    pthread_mutex_lock(&r->h->lock);
    while (r->h->head < r->h->tail) {
        uint64_t off = r->h->head % r->cap;
        struct ring_rec rh;
        memcpy(&rh, r->data + off, sizeof rh);
        if (rh.kind == RING_KIND_PAD) { r->h->head += rh.len; continue; }
        *rec = rh;
        *payload = r->data + off + sizeof rh;
        r->h->head += rh.len;
        got = 1;
        break;
    }
    pthread_mutex_unlock(&r->h->lock);
    if (got) __atomic_add_fetch(&fakeib_ring_records_in, 1, __ATOMIC_RELAXED);
    return got;
}

/* Pending RDMA READs: the response arrives on the reverse ring later, so the
 * WR's completion cannot be posted at post_send time. */
struct pending_read {
    int used;
    uint64_t tag;
    struct fake_qp *q;
    uint64_t wr_id;
    struct ibv_sge sge[8];
    int nsge;
    uint64_t ready_ns;
};
static struct pending_read g_reads[4096];
static pthread_mutex_t g_reads_lock = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_read_seq;

static uint64_t read_register(struct fake_qp *q, struct ibv_send_wr *w, uint64_t ready_ns) {
    uint64_t tag = ((uint64_t)q->qp.qp_num << 32) | (++g_read_seq & 0xffffffffull);
    pthread_mutex_lock(&g_reads_lock);
    for (int i = 0; i < 4096; i++) {
        if (g_reads[i].used) continue;
        g_reads[i] = (struct pending_read){.used = 1, .tag = tag, .q = q,
                                           .wr_id = w->wr_id,
                                           .nsge = w->num_sge < 8 ? w->num_sge : 8,
                                           .ready_ns = ready_ns};
        for (int j = 0; j < g_reads[i].nsge; j++) g_reads[i].sge[j] = w->sg_list[j];
        pthread_mutex_unlock(&g_reads_lock);
        return tag;
    }
    pthread_mutex_unlock(&g_reads_lock);
    return 0;
}

static void read_complete(uint64_t tag, const char *payload, uint32_t nbytes,
                          uint32_t status, uint64_t ready_ns) {
    struct ibv_wc wc;
    memset(&wc, 0, sizeof wc);
    struct fake_qp *q = NULL;
    pthread_mutex_lock(&g_reads_lock);
    for (int i = 0; i < 4096; i++) {
        if (!g_reads[i].used || g_reads[i].tag != tag) continue;
        struct pending_read *pr = &g_reads[i];
        q = pr->q;
        wc.wr_id = pr->wr_id;
        wc.qp_num = q->qp.qp_num;
        wc.opcode = IBV_WC_RDMA_READ;
        if (status == 0) {
            wc.status = IBV_WC_SUCCESS;
            wc.byte_len = (uint32_t)flat_to_sg(payload, nbytes, pr->sge, pr->nsge);
            __atomic_add_fetch(&fakeib_bytes_read, nbytes, __ATOMIC_RELAXED);
            __atomic_add_fetch(&fakeib_reads, 1, __ATOMIC_RELAXED);
        } else {
            wc.status = IBV_WC_REM_ACCESS_ERR;
        }
        pr->used = 0;
        break;
    }
    pthread_mutex_unlock(&g_reads_lock);
    if (q) cq_push(q->send_cq, &wc, ready_ns);
}

/* Drain every ring addressed to this rank.  Called at the top of poll_cq, which
 * is where NCCL's progress loop already is. */
/* A receiver cannot see an inbound ring until it opens that file: rings are
 * created lazily by whoever touches them first.  FAKEIB_NRANKS (default 2)
 * tells every rank which pair-rings to open in BOTH directions up front, so a
 * rank that only ever receives still has the ring in its pump list. */
static pthread_once_t g_rings_once = PTHREAD_ONCE_INIT;
static void rings_init(void) {
    const char *e = getenv("FAKEIB_NRANKS");
    int n = (e && atoi(e) > 0) ? atoi(e) : 2;
    int me = my_rank();
    for (int r = 0; r < n; r++) {
        if (r == me) continue;
        ring_open(me, r);
        ring_open(r, me);
    }
}
/* gpu-sim-7q3w.5: inbound progress must not require the RECEIVER to call
 * poll_cq first.  Measured, not assumed: NCCL's IB p2p send setup spins on a
 * host-memory FIFO that the peer's 64-byte RDMA WRITE must fill; the old model
 * only copied that write into the receiver's memory inside poll_cq, which the
 * sender never called because it was spinning on the FIFO.  Both ranks waited
 * on each other (rank1 sat in poll_cq with rings out=49 in=48 forever).  On real
 * hardware the NIC writes the buffer asynchronously, so the model gets a pump
 * thread too.  All draining is serialised by g_pump_lock, which keeps inbound
 * delivery single-threaded (the RQ consume order stays deterministic). */
static pthread_mutex_t g_pump_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_pump_once = PTHREAD_ONCE_INIT;
static void pump_rings(void);

static void *pump_thread_fn(void *arg) {
    (void)arg;
    const char *e = getenv("FAKEIB_PUMP_US");
    long us = (e && atol(e) > 0) ? atol(e) : 50;   /* added delivery latency bound */
    struct timespec ts = {0, us * 1000};
    for (;;) {
        pump_rings();
        nanosleep(&ts, NULL);
    }
    return NULL;
}

static void pump_thread_start(void) {
    if (getenv("FAKEIB_NO_PUMP_THREAD")) return;   /* A/B: off restores 7q3w.3 */
    pthread_t t;
    if (pthread_create(&t, NULL, pump_thread_fn, NULL) == 0) pthread_detach(t);
}

/* Open the pair rings and start the pump thread WITHOUT waiting for a poll_cq.
 * Measured (gpu-sim-7q3w.5 control, phase B): a p2p-alone SENDER never calls
 * ibv_poll_cq -- it spins on the host FIFO the peer's RDMA WRITE must fill -- so
 * starting the pump lazily from pump_rings() left rank0 with no inbound progress
 * at all (it logged not one FAKEIB line) and the 64-byte connection write sat in
 * the ring forever.  Every rank opens its device, so that is the eager hook. */
static void fakeib_pump_ensure(void) {
    pthread_once(&g_rings_once, rings_init);
    pthread_once(&g_pump_once, pump_thread_start);
}

static void pump_rings(void) {
    fakeib_pump_ensure();
    pthread_mutex_lock(&g_pump_lock);
    if (getenv("FAKEIB_TRACE")) {
        static int n_trace;
        if (n_trace++ < 3)
            fprintf(stderr, "FAKEIB[%d] pump rings=%d\n", my_rank(), g_nrings);
    }
    int me = my_rank();
    for (int i = 0; i < g_nrings; i++) {
        struct ring *r = g_rings[i];
        if (r->to != me) continue;
        struct ring_rec rec;
        const char *payload = NULL;
        while (ring_get(r, &rec, &payload)) {
            switch (rec.kind) {
            case RING_KIND_WRITE:
            case RING_KIND_WRITE_IMM:
                deliver_inbound(rec.dest_qpn, rec.kind == RING_KIND_WRITE_IMM ?
                                IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE,
                                rec.wr_id, rec.addr, rec.rkey, rec.imm,
                                payload, rec.nbytes, rec.ready_ns);
                break;
            case RING_KIND_SEND:
                deliver_inbound(rec.dest_qpn, IBV_WR_SEND, rec.wr_id, 0, 0, rec.imm,
                                payload, rec.nbytes, rec.ready_ns);
                break;
            case RING_KIND_READ_REQ: {
                void *src = remote_to_local(rec.addr, rec.rkey, rec.nbytes);
                struct ring_rec resp = rec;
                resp.kind = RING_KIND_READ_RESP;
                resp.status = src ? 0 : 2;
                if (!src) resp.nbytes = 0;
                struct ring *back = ring_open(me, r->from);
                ring_put(back, &resp, src ? src : "");
                break;
            }
            case RING_KIND_READ_RESP:
                read_complete(rec.tag, payload, rec.nbytes, rec.status, rec.ready_ns);
                break;
            default:
                break;
            }
        }
    }
    pthread_mutex_unlock(&g_pump_lock);
}

/* One WR against a peer in another process. */
static void ring_send_wr(struct fake_qp *q, struct ibv_send_wr *w, uint64_t ready_ns) {
    int signalled = (w->send_flags & IBV_SEND_SIGNALED) || q->sq_sig_all;
    size_t bytes = wr_payload_bytes(w->sg_list, w->num_sge);
    struct ring *r = ring_open(my_rank(), q->peer_rank);
    struct ring_rec rec;
    memset(&rec, 0, sizeof rec);
    rec.wr_id = w->wr_id;
    rec.dest_qpn = q->peer_qpn;
    rec.ready_ns = ready_ns;
    rec.nbytes = (uint32_t)bytes;
    int rc = 0;

    if (w->opcode == IBV_WR_RDMA_WRITE || w->opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
        rec.kind = (w->opcode == IBV_WR_RDMA_WRITE_WITH_IMM) ? RING_KIND_WRITE_IMM : RING_KIND_WRITE;
        rec.addr = w->wr.rdma.remote_addr;
        rec.rkey = w->wr.rdma.rkey;
        rec.imm = w->imm_data;
        void *flat = bytes ? malloc(bytes) : NULL;
        size_t got = flat ? sg_to_flat(w->sg_list, w->num_sge, flat, bytes) : 0;
        rec.nbytes = (uint32_t)got;
        rc = ring_put(r, &rec, flat);
        free(flat);
        if (signalled) {
            struct ibv_wc wc;
            memset(&wc, 0, sizeof wc);
            wc.wr_id = w->wr_id;
            wc.qp_num = q->qp.qp_num;
            wc.opcode = IBV_WC_RDMA_WRITE;
            wc.status = rc ? IBV_WC_LOC_QP_OP_ERR : IBV_WC_SUCCESS;
            cq_push(q->send_cq, &wc, ready_ns);
        }
        if (!rc) {
            __atomic_add_fetch(&fakeib_bytes_written, got, __ATOMIC_RELAXED);
            __atomic_add_fetch(&fakeib_writes, 1, __ATOMIC_RELAXED);
        }
        return;
    }
    if (w->opcode == IBV_WR_SEND || w->opcode == IBV_WR_SEND_WITH_IMM) {
        rec.kind = RING_KIND_SEND;
        rec.imm = w->imm_data;
        void *flat = bytes ? malloc(bytes) : NULL;
        size_t got = flat ? sg_to_flat(w->sg_list, w->num_sge, flat, bytes) : 0;
        rec.nbytes = (uint32_t)got;
        rc = ring_put(r, &rec, flat);
        free(flat);
        __atomic_add_fetch(&fakeib_sends, 1, __ATOMIC_RELAXED);
        if (signalled) {
            struct ibv_wc wc;
            memset(&wc, 0, sizeof wc);
            wc.wr_id = w->wr_id;
            wc.qp_num = q->qp.qp_num;
            wc.opcode = IBV_WC_SEND;
            wc.byte_len = (uint32_t)got;
            wc.status = rc ? IBV_WC_LOC_QP_OP_ERR : IBV_WC_SUCCESS;
            cq_push(q->send_cq, &wc, ready_ns);
        }
        return;
    }
    if (w->opcode == IBV_WR_RDMA_READ) {
        uint64_t tag = read_register(q, w, ready_ns);
        if (!tag) {
            struct ibv_wc wc;
            memset(&wc, 0, sizeof wc);
            wc.wr_id = w->wr_id; wc.qp_num = q->qp.qp_num;
            wc.opcode = IBV_WC_RDMA_READ; wc.status = IBV_WC_LOC_QP_OP_ERR;
            cq_push(q->send_cq, &wc, ready_ns);
            return;
        }
        rec.kind = RING_KIND_READ_REQ;
        rec.tag = tag;
        rec.addr = w->wr.rdma.remote_addr;
        rec.rkey = w->wr.rdma.rkey;
        rec.nbytes = (uint32_t)bytes;
        if (ring_put(r, &rec, NULL) != 0) read_complete(tag, NULL, 0, 2, ready_ns);
        return;
    }
    /* Unsupported opcode: complete with a local error rather than hang. */
    if (signalled) {
        struct ibv_wc wc;
        memset(&wc, 0, sizeof wc);
        wc.wr_id = w->wr_id; wc.qp_num = q->qp.qp_num;
        wc.status = IBV_WC_LOC_QP_OP_ERR;
        cq_push(q->send_cq, &wc, ready_ns);
    }
}
