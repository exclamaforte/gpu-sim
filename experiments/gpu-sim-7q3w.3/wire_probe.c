/* gpu-sim-7q3w.3: cross-process wire probe -- two real processes, one shim.
 *
 * This is the wire's own test, without NCCL and without sysfs spoofing.  It
 * forks two ranks, exchanges (qpn, rkey, addr) over pipes exactly the way NCCL
 * exchanges them, and then drives:
 *   - RDMA WRITE           (bytes land in the peer's registered buffer)
 *   - RDMA WRITE_WITH_IMM  (bytes land AND the peer's posted recv WR completes
 *                           with IBV_WC_RECV_RDMA_WITH_IMM and the imm data)
 *   - RDMA READ            (peer's bytes come back into the local sge)
 *   - the modelled delay ACROSS processes (a 20 ms latency arm, timed on the
 *     sender's own completion; the ring carries an absolute ready_ns and both
 *     processes share CLOCK_MONOTONIC)
 *
 * A stale ring file is not a peer: every run creates a fresh FAKEIB_RING_DIR.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "fakeib_api.h"

static int failures;
#define CHECK(cond, ...) do {                                              \
    if (!(cond)) { fprintf(stderr, "rank%d FAIL %s:%d: ", my_rank_,       \
                           __FILE__, __LINE__);                           \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n");    \
                   failures++; }                                           \
} while (0)

static int my_rank_;

/* WIRE_TRACE=1: stderr phases, unbuffered, so a hang names the step instead of
 * being a two-line log with no ordering.  A hang here cannot be reproduced on
 * demand; the trace is what makes the next one diagnosable. */
#define TRACE(...) do {                                                    \
    if (getenv("WIRE_TRACE")) {                                            \
        fprintf(stderr, "[r%d] ", my_rank_);                               \
        fprintf(stderr, __VA_ARGS__);                                     \
        fprintf(stderr, "\n");                                            \
    }                                                                      \
} while (0)

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static int poll_until(struct ibv_cq *cq, int n, struct ibv_wc *wc, uint64_t timeout_ms) {
    uint64_t deadline = now_ns() + timeout_ms * 1000000ull;
    int got = 0;
    while (got < n && now_ns() < deadline) {
        int r = probe_poll_cq(cq, n - got, wc + got);
        if (r < 0) return got;
        got += r;
        if (got < n) { struct timespec ts = {0, 100000}; nanosleep(&ts, NULL); }
    }
    return got;
}
static int read_full(int fd, void *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t r = read(fd, (char *)buf + off, len - off);
        if (r <= 0) return -1;
        off += (size_t)r;
    }
    return 0;
}
static int write_full(int fd, const void *buf, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t r = write(fd, (const char *)buf + off, len - off);
        if (r <= 0) return -1;
        off += (size_t)r;
    }
    return 0;
}

struct handshake {
    uint32_t qpn;
    uint32_t rkey;
    uint64_t addr;
    uint32_t len;
    uint32_t pad;
};

static struct ibv_qp *make_qp(struct ibv_pd *pd, struct ibv_cq *scq, struct ibv_cq *rcq) {
    struct ibv_qp_init_attr qia;
    memset(&qia, 0, sizeof qia);
    qia.send_cq = scq;
    qia.recv_cq = rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = 64;
    qia.cap.max_recv_wr = 64;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    struct ibv_qp *qp = ibv_create_qp(pd, &qia);
    if (!qp) return NULL;
    struct ibv_qp_attr qa;
    memset(&qa, 0, sizeof qa);
    qa.qp_state = IBV_QPS_INIT;
    qa.pkey_index = 0;
    qa.port_num = 1;
    qa.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    if (ibv_modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                IBV_QP_ACCESS_FLAGS) != 0) return NULL;
    return qp;
}

static int connect_qp(struct ibv_qp *qp, const struct handshake *peer, int peer_rank) {
    struct ibv_qp_attr qa;
    memset(&qa, 0, sizeof qa);
    qa.qp_state = IBV_QPS_RTR;
    qa.path_mtu = IBV_MTU_4096;
    qa.dest_qp_num = peer->qpn;
    qa.rq_psn = 1;
    qa.ah_attr.is_global = 1;
    qa.ah_attr.grh.dgid.raw[15] = (uint8_t)peer_rank;
    qa.ah_attr.port_num = 1;
    if (ibv_modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0) return -1;
    memset(&qa, 0, sizeof qa);
    qa.qp_state = IBV_QPS_RTS;
    qa.sq_psn = 1;
    if (ibv_modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) return -1;
    return 0;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    /* Fresh ring dir for this run: stale rings are a different peer, and the
     * two ranks must agree on the path (set before fork).  GPU_SIM_ROOT is the
     * worktree root; never a hardcoded $HOME. */
    char ringdir[256];
    const char *root = getenv("GPU_SIM_ROOT");
    if (!root || !*root) root = ".";
    snprintf(ringdir, sizeof ringdir, "%s/.lab/tmp/7q3w3-wire-XXXXXX", root);
    if (!mkdtemp(ringdir)) { perror("mkdtemp"); return 1; }
    setenv("FAKEIB_RING_DIR", ringdir, 1);

    /* One pipe PER DIRECTION.  A single pipe gave both ranks a read end on the
     * same byte stream, and rank1's d-wait loop (non-blocking) could read the
     * 'i' it had just written, leaving rank0 blocked forever on a read that
     * could never be satisfied.  Two pipes, one reader each, no stealing -- and
     * a dead peer now shows up as EOF instead of a hang. */
    int to_child[2], to_parent[2];
    int acks_up[2], ack_down[2];   /* rank1 -> rank0 'a','i'; rank0 -> rank1 'd' */
    if (pipe(to_child) || pipe(to_parent) || pipe(acks_up) || pipe(ack_down)) {
        perror("pipe");
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    int is_rank0_ = (pid != 0);                       /* the PARENT is rank 0 */
    my_rank_ = is_rank0_ ? 0 : 1;
    char rbuf[16];
    snprintf(rbuf, sizeof rbuf, "%d", my_rank_);
    setenv("FAKEIB_RANK", rbuf, 1);
    setenv("FAKEIB_NRANKS", "2", 1);

    int wfd = is_rank0_ ? to_child[1] : to_parent[1];  /* my handshake out */
    int rfd = is_rank0_ ? to_parent[0] : to_child[0];  /* peer handshake in */
    int ackr = is_rank0_ ? acks_up[0] : ack_down[0];
    int ackw = is_rank0_ ? ack_down[1] : acks_up[1];
    /* Drop the ends this process must never touch (a dead peer is then EOF,
     * not a second reader that swallows our own bytes). */
    if (is_rank0_) { close(acks_up[1]); close(ack_down[0]); }
    else           { close(acks_up[0]); close(ack_down[1]); }

    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    CHECK(list && num == 1, "device list num=%d", num);
    struct ibv_context *ctx = ibv_open_device(list[0]);
    CHECK(ctx != NULL, "open_device");
    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    CHECK(pd != NULL, "alloc_pd");
    char *buf = aligned_alloc(4096, 8192);
    memset(buf, is_rank0_ ? 0x33 : 0x77, 8192);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, 8192, IBV_ACCESS_LOCAL_WRITE |
                                   IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    CHECK(mr != NULL, "reg_mr");
    struct ibv_cq *scq = ibv_create_cq(ctx, 64, NULL, NULL, 0);
    struct ibv_cq *rcq = ibv_create_cq(ctx, 64, NULL, NULL, 0);
    CHECK(scq && rcq, "create_cq");
    struct ibv_qp *qp = make_qp(pd, scq, rcq);
    CHECK(qp != NULL, "create_qp/INIT");

    struct handshake mine = {.qpn = qp->qp_num, .rkey = mr->rkey,
                             .addr = (uint64_t)(uintptr_t)buf, .len = 8192};
    struct handshake peer;
    if (write_full(wfd, &mine, sizeof mine) != 0) { perror("hs write"); return 1; }
    if (read_full(rfd, &peer, sizeof peer) != 0) { perror("hs read"); return 1; }
    CHECK(connect_qp(qp, &peer, is_rank0_ ? 1 : 0) == 0, "connect_qp");
    TRACE("connected, peer qpn=%u addr=0x%llx", peer.qpn,
          (unsigned long long)peer.addr);

    char ackbyte = 0;
    struct ibv_wc wc[4];
    struct ibv_send_wr *badwr = NULL;

    if (is_rank0_) {
        /* ---------------- rank 0: writer / reader ---------------- */
        char payload[4096];
        for (int i = 0; i < (int)sizeof payload; i++) payload[i] = (char)(i * 11 + 5);
        struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)payload,
                              .length = sizeof payload, .lkey = mr->lkey};
        struct ibv_send_wr wr;
        memset(&wr, 0, sizeof wr);
        wr.wr_id = 0x1001;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.wr.rdma.remote_addr = peer.addr;
        wr.wr.rdma.rkey = peer.rkey;

        /* modelled latency across processes, timed on the sender's WC */
        setenv("FAKEIB_LATENCY_US", "20000", 1);
        uint64_t t0 = now_ns();
        CHECK(probe_post_send(qp, &wr, &badwr) == 0, "post WRITE");
        TRACE("posted WRITE");
        CHECK(poll_until(scq, 1, wc, 2000) == 1, "WRITE completion");
        TRACE("WRITE wc, waiting ack a");
        uint64_t dt = now_ns() - t0;
        CHECK(dt >= 20000000ull, "WRITE completion at %llu ns, want >= 20 ms",
              (unsigned long long)dt);
        printf("rank0: cross-process WRITE completion after %llu ns (latency arm)\n",
               (unsigned long long)dt);
        setenv("FAKEIB_LATENCY_US", "1.2", 1);
        CHECK(read_full(ackr, &ackbyte, 1) == 0 && ackbyte == 'a', "rank1 saw WRITE bytes");
        TRACE("got ack '%c' (WRITE bytes)", ackbyte);

        memset(&wr, 0, sizeof wr);
        wr.wr_id = 0x1002;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.imm_data = 0xfeedface;
        wr.wr.rdma.remote_addr = peer.addr + 4096;
        wr.wr.rdma.rkey = peer.rkey;
        CHECK(probe_post_send(qp, &wr, &badwr) == 0, "post WRITE_WITH_IMM");
        TRACE("posted WRITE_WITH_IMM");
        CHECK(poll_until(scq, 1, wc, 2000) == 1, "WRITE_WITH_IMM send completion");
        TRACE("IMM wc, waiting ack i");
        CHECK(read_full(ackr, &ackbyte, 1) == 0 && ackbyte == 'i', "rank1 saw IMM");
        TRACE("got ack '%c' (IMM)", ackbyte);

        char got[4096];
        memset(got, 0, sizeof got);
        struct ibv_sge rsge = {.addr = (uint64_t)(uintptr_t)got,
                               .length = sizeof got, .lkey = mr->lkey};
        memset(&wr, 0, sizeof wr);
        wr.wr_id = 0x1003;
        wr.opcode = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.sg_list = &rsge;
        wr.num_sge = 1;
        wr.wr.rdma.remote_addr = peer.addr + 4096;
        wr.wr.rdma.rkey = peer.rkey;
        CHECK(probe_post_send(qp, &wr, &badwr) == 0, "post READ");
        TRACE("posted READ");
        CHECK(poll_until(scq, 1, wc, 2000) == 1, "READ completion");
        TRACE("READ wc (before checks)");
        CHECK(wc[0].opcode == IBV_WC_RDMA_READ, "read opcode %d", wc[0].opcode);
        CHECK(wc[0].byte_len == sizeof got, "read byte_len %u", wc[0].byte_len);
        CHECK(memcmp(got, payload, sizeof payload) == 0, "READ returned the peer's bytes");
        ackbyte = 'd';
        write_full(ackw, &ackbyte, 1);
    } else {
        /* ---------------- rank 1: target ---------------- */
        struct ibv_recv_wr rwr;
        memset(&rwr, 0, sizeof rwr);
        rwr.wr_id = 0x2001;
        struct ibv_sge rsge = {.addr = (uint64_t)(uintptr_t)buf, .length = 4096,
                               .lkey = mr->lkey};
        rwr.sg_list = &rsge;
        rwr.num_sge = 1;
        struct ibv_recv_wr *badrwr = NULL;
        CHECK(probe_post_recv(qp, &rwr, &badrwr) == 0, "post recv for IMM");

        /* RDMA WRITE has no receiver completion: observe the buffer. */
        uint64_t deadline = now_ns() + 2000000000ull;
        int seen = 0;
        while (now_ns() < deadline) {
            if ((unsigned char)buf[0] == (unsigned char)(0 * 11 + 5) &&
                (unsigned char)buf[4095] == (unsigned char)(4095 * 11 + 5)) { seen = 1; break; }
            struct timespec ts = {0, 200000}; nanosleep(&ts, NULL);
            probe_poll_cq(rcq, 4, wc);   /* also drives pump_rings() */
        }
        CHECK(seen, "peer's RDMA WRITE bytes never landed");
        TRACE("peer bytes landed, writing a");
        ackbyte = 'a';
        TRACE("wrote ack a");
        write_full(ackw, &ackbyte, 1);

        CHECK(poll_until(rcq, 1, wc, 2000) == 1, "IMM receiver completion");
        TRACE("IMM seen, writing i");
        CHECK(wc[0].wr_id == 0x2001, "IMM wr_id 0x%llx", (unsigned long long)wc[0].wr_id);
        CHECK(wc[0].opcode == IBV_WC_RECV_RDMA_WITH_IMM, "IMM opcode %d", wc[0].opcode);
        CHECK(wc[0].imm_data == 0xfeedface, "IMM data 0x%x", wc[0].imm_data);
        CHECK(buf[4096] == (char)(0 * 11 + 5), "IMM write landed in the right MR range");
        ackbyte = 'i';
        TRACE("wrote ack i, waiting d");
        write_full(ackw, &ackbyte, 1);

        /* rank0's READ needs us to keep pumping the rings while we wait for
         * its 'd' ack: a blocking read here would deadlock both ranks. */
        int fl = fcntl(ackr, F_GETFL, 0);
        fcntl(ackr, F_SETFL, fl | O_NONBLOCK);
        uint64_t dl = now_ns() + 2000000000ull;
        int done = 0;
        int iters = 0;
        while (now_ns() < dl) {
            if (++iters % 2000 == 0) TRACE("waiting ack d, iter=%d", iters);
            probe_poll_cq(rcq, 4, wc);     /* drives pump_rings() */
            if (read(ackr, &ackbyte, 1) == 1 && ackbyte == 'd') { done = 1; break; }
            struct timespec ts = {0, 100000};
            nanosleep(&ts, NULL);
        }
        CHECK(done, "rank0 finished READ");
        TRACE("ack d %s after %d iters", done ? "seen" : "MISSED", iters);
    }

    ibv_destroy_qp(qp);
    ibv_destroy_cq(rcq);
    ibv_destroy_cq(scq);
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(list);

    if (is_rank0_) {
        int st = 0;
        waitpid(pid, &st, 0);
        CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "child exit status %d", st);
        /* Leave no stale ring behind: 2x64 MiB per run, and a stale ring is a
         * peer that no longer exists (the fs is metadata-tight on this box). */
        char p[512];
        for (int i = 0; i < 2; i++) {
            snprintf(p, sizeof p, "%s/ring-%d-%d.bin", ringdir, i, 1 - i);
            unlink(p);
        }
        rmdir(ringdir);
        if (failures) { printf("WIRE FAILED: %d failure(s)\n", failures); return 1; }
        printf("WIRE OK\n");
        TRACE("parent: WIRE OK");
        return 0;
    }
    return failures ? 1 : 0;
    TRACE("exiting, failures=%d", failures);
}
