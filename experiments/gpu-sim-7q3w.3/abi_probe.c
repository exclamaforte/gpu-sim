/* gpu-sim-7q3w.3: ABI + semantics probe for the libibverbs shim.
 *
 * This is the shim's own test: it does not need NCCL.  It answers the bead's
 * contract items that can be answered without a peer process:
 *   - every symbol NCCL dlsyms is exported (run.sh checks the 32-name list)
 *   - QP state machine ACCEPTS INIT->RTR->RTS and REFUSES an impossible edge
 *   - one posted send => exactly one completion, correct wr_id/opcode/byte_len
 *   - an empty CQ polls 0 (not -1/errno)
 *   - RDMA WRITE / WRITE_WITH_IMM / READ move the bytes and complete correctly
 *   - the completion time is the modelled wire (latency + bytes/bandwidth)
 *   - a comp-channel event fires on the armed CQ
 *   - the dmabuf (GDR) registration counter is non-zero
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>   /* memfd_create, for the dmabuf registration test */
#include <time.h>
#include <unistd.h>

#include "fakeib_api.h"

static int failures;
#define CHECK(cond, ...) do {                                              \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);    \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n");    \
                   failures++; }                                           \
} while (0)

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* poll until n completions or timeout; returns count, fills wc[] */
static int poll_until(struct ibv_cq *cq, int n, struct ibv_wc *wc, uint64_t timeout_ms) {
    uint64_t deadline = now_ns() + timeout_ms * 1000000ull;
    int got = 0;
    while (got < n && now_ns() < deadline) {
        int r = probe_poll_cq(cq, n - got, wc + got);
        CHECK(r >= 0, "poll_cq returned %d (must never be negative)", r);
        if (r < 0) return got;
        got += r;
    }
    return got;
}

int main(void) {
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    CHECK(list && num == 1, "device list: got %p num=%d", (void *)list, num);
    if (!list || num < 1) return 1;
    printf("device: %s\n", ibv_get_device_name(list[0]));

    struct ibv_context *ctx = ibv_open_device(list[0]);
    CHECK(ctx != NULL, "open_device");
    if (!ctx) return 1;

    struct ibv_device_attr da;
    CHECK(ibv_query_device(ctx, &da) == 0, "query_device");
    CHECK(da.vendor_id == 0x02c9, "vendor_id 0x%x", da.vendor_id);
    CHECK(da.max_sge >= 1 && da.max_qp_wr >= 128 && da.max_cqe >= 64,
          "caps sge=%d wr=%d cqe=%d", da.max_sge, da.max_qp_wr, da.max_cqe);

    struct ibv_port_attr pa;
    CHECK(ibv_query_port(ctx, 1, &pa) == 0, "query_port");
    CHECK(pa.state == IBV_PORT_ACTIVE, "port state %d", pa.state);
    /* NCCL skips a port that is neither IB nor Ethernet (net_ib/init.cc:511);
     * leaving link_layer unset sent NCCL to Socket with everything else right,
     * and only the GPU arm noticed.  Assert it here so the CPU stages do too. */
    CHECK(pa.link_layer == IBV_LINK_LAYER_INFINIBAND, "link_layer %d", pa.link_layer);
    CHECK(pa.active_speed == 128 && pa.active_width == 2,
          "NDR 4x expected (speed 128, width 2), got %d/%d", pa.active_speed,
          pa.active_width);
    CHECK(pa.active_mtu == IBV_MTU_4096, "mtu %d", pa.active_mtu);
    CHECK(pa.gid_tbl_len == 1, "gid_tbl_len %d", pa.gid_tbl_len);

    union ibv_gid gid;
    CHECK(ibv_query_gid(ctx, 1, 0, &gid) == 0, "query_gid");
    CHECK(gid.raw[0] == 0xfe && gid.raw[1] == 0x80, "gid prefix");
    uint16_t pkey = 0;
    CHECK(ibv_query_pkey(ctx, 1, 0, &pkey) == 0 && pkey == 0xffff, "pkey");

    uint64_t speed = 0;
    int (*qps_fn)(struct ibv_context *, uint8_t, uint64_t *) =
        (int (*)(struct ibv_context *, uint8_t, uint64_t *))dlsym(RTLD_DEFAULT, "ibv_query_port_speed");
    CHECK(qps_fn != NULL, "ibv_query_port_speed exported");
    if (qps_fn) {
        CHECK(qps_fn(ctx, 1, &speed) == 0, "query_port_speed");
        CHECK(speed == 4000, "speed in 100Mb/s units: got %llu want 4000", (unsigned long long)speed);
    }

    struct ibv_pd *pd = ibv_alloc_pd(ctx);
    CHECK(pd != NULL, "alloc_pd");
    char *buf = aligned_alloc(4096, 1 << 20);
    char *rbuf = aligned_alloc(4096, 1 << 20);
    CHECK(buf && rbuf, "alloc buffers");
    memset(buf, 0x5a, 1 << 20);
    memset(rbuf, 0, 1 << 20);
    struct ibv_mr *mr = ibv_reg_mr(pd, buf, 1 << 20, IBV_ACCESS_LOCAL_WRITE);
    struct ibv_mr *rmr = ibv_reg_mr(pd, rbuf, 1 << 20,
                                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ);
    CHECK(mr && rmr, "reg_mr");
    CHECK(mr && mr->addr == buf && mr->length == (1 << 20), "mr fields");
    CHECK(mr && rmr && mr->rkey != rmr->rkey, "distinct rkeys");

    struct ibv_comp_channel *chan = ibv_create_comp_channel(ctx);
    CHECK(chan && chan->fd >= 0, "comp channel with a readable fd");
    struct ibv_cq *scq = ibv_create_cq(ctx, 64, NULL, NULL, 0);
    /* qp1 gets its OWN recv CQ: the SEND arms below receive on qp1, and rcq
     * is reserved for qp2 so the IMM/event arms see only their own WC. */
    struct ibv_cq *rcq1 = ibv_create_cq(ctx, 64, NULL, NULL, 0);
    struct ibv_cq *rcq = ibv_create_cq(ctx, 64, NULL, chan, 0);
    CHECK(scq && rcq1 && rcq, "create_cq");

    struct ibv_qp_init_attr qia;
    memset(&qia, 0, sizeof qia);
    qia.send_cq = scq;
    qia.recv_cq = rcq1;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = 128;
    qia.cap.max_recv_wr = 128;
    qia.cap.max_send_sge = 1;
    qia.cap.max_recv_sge = 1;
    struct ibv_qp *qp = ibv_create_qp(pd, &qia);
    CHECK(qp != NULL, "create_qp");
    CHECK(qia.cap.max_send_wr == 128, "cap echoed: %u", qia.cap.max_send_wr);

    struct ibv_qp_attr qa;
    memset(&qa, 0, sizeof qa);
    qa.qp_state = IBV_QPS_INIT;
    qa.pkey_index = 0;
    qa.port_num = 1;
    qa.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    CHECK(ibv_modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
          "RESET->INIT");
    memset(&qa, 0, sizeof qa);
    qa.qp_state = IBV_QPS_RTR;
    qa.path_mtu = IBV_MTU_4096;
    qa.dest_qp_num = qp->qp_num;
    qa.rq_psn = 7;
    qa.ah_attr.is_global = 1;
    qa.ah_attr.grh.dgid.raw[15] = 0;   /* ourselves: loopback */
    qa.ah_attr.port_num = 1;
    CHECK(ibv_modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) == 0, "INIT->RTR");
    memset(&qa, 0, sizeof qa);
    qa.qp_state = IBV_QPS_RTS;
    qa.timeout = 14; qa.retry_cnt = 7; qa.rnr_retry = 7;
    qa.sq_psn = 7; qa.max_rd_atomic = 1; qa.max_dest_rd_atomic = 1;
    CHECK(ibv_modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_MAX_DEST_RD_ATOMIC) == 0,
          "RTR->RTS");
    /* impossible edge must be REFUSED, not accepted */
    int bad = ibv_modify_qp(qp, &qa, IBV_QP_STATE);   /* RTS->RTS */
    qa.qp_state = IBV_QPS_INIT;
    bad = ibv_modify_qp(qp, &qa, IBV_QP_STATE);       /* RTS->INIT */
    CHECK(bad == EINVAL, "RTS->INIT must be refused, got %d", bad);

    /* empty CQ: 0, never negative */
    struct ibv_wc wc[8];
    CHECK(probe_poll_cq(scq, 8, wc) == 0, "empty CQ polls 0");

    /* Post three recv WRs on qp so the SEND arms below land in a receive
     * queue.  A SEND with no posted recv is RNR and deliberately does not
     * complete on the sender (real IB semantics), so this is required. */
    struct ibv_recv_wr rwr0[3];
    struct ibv_sge rsge0[3];
    struct ibv_recv_wr *badrwr = NULL;
    for (int i = 0; i < 3; i++) {
        memset(&rwr0[i], 0, sizeof rwr0[i]);
        rsge0[i] = (struct ibv_sge){.addr = (uint64_t)(uintptr_t)rbuf + (size_t)i * 64,
                                    .length = 64, .lkey = rmr->lkey};
        rwr0[i].wr_id = 0x9000 + (uint64_t)i;
        rwr0[i].sg_list = &rsge0[i];
        rwr0[i].num_sge = 1;
        CHECK(probe_post_recv(qp, &rwr0[i], &badrwr) == 0, "post_recv %d", i);
    }

    /* --- modelled latency gates the completion -------------------------- */
    setenv("FAKEIB_LATENCY_US", "40000", 1);   /* 40 ms, far above noise */
    struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)buf, .length = 64, .lkey = mr->lkey};
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof wr);
    wr.wr_id = 0x1111;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    struct ibv_send_wr *badwr = NULL;
    uint64_t t0 = now_ns();
    CHECK(probe_post_send(qp, &wr, &badwr) == 0, "post_send");
    CHECK(probe_poll_cq(scq, 8, wc) == 0, "completion must NOT be ready before the model says so");
    int got = poll_until(scq, 1, wc, 1000);
    uint64_t dt = now_ns() - t0;
    CHECK(got == 1, "latency arm: got %d", got);
    CHECK(dt >= 40000000ull, "latency arm took %llu ns, want >= 40 ms", (unsigned long long)dt);
    if (got == 1) {
        CHECK(wc[0].wr_id == 0x1111, "wr_id 0x%llx", (unsigned long long)wc[0].wr_id);
        CHECK(wc[0].opcode == IBV_WC_SEND, "opcode %d", wc[0].opcode);
        CHECK(wc[0].byte_len == 64, "byte_len %u", wc[0].byte_len);
    }

    /* --- bandwidth: 2x line rate => half the modelled charge --------------
     * The assertion is on the shim's OWN charge (deterministic), not on wall
     * clock: a 1 MiB arm takes ~839 us at 10 Gb/s, and one scheduler slice on
     * a loaded box reordered a wall-clock ratio to 32x while the model was
     * correct.  The latency arm above already proves poll waits for the wire;
     * this one proves the bytes/bandwidth term scales exactly.  The wall-clock
     * numbers are printed, with only a gross sanity bound. */
    setenv("FAKEIB_LATENCY_US", "0", 1);
    size_t big = 1 << 20;
    sge.length = (uint32_t)big;
    uint64_t *charge = (uint64_t *)dlsym(RTLD_DEFAULT, "fakeib_last_charge_ns");
    uint64_t *charge_bytes = (uint64_t *)dlsym(RTLD_DEFAULT, "fakeib_last_charge_bytes");
    CHECK(charge && charge_bytes, "last-charge counters exported");
    setenv("FAKEIB_BW_GBPS", "10", 1);
    wr.wr_id = 0x2222;
    t0 = now_ns();
    CHECK(probe_post_send(qp, &wr, &badwr) == 0, "post_send 10G");
    uint64_t charge10 = charge ? *charge : 0;
    uint64_t cbytes10 = charge_bytes ? *charge_bytes : 0;
    CHECK(poll_until(scq, 1, wc, 5000) == 1, "10G completion");
    uint64_t slow = now_ns() - t0;
    setenv("FAKEIB_BW_GBPS", "20", 1);
    wr.wr_id = 0x3333;
    t0 = now_ns();
    CHECK(probe_post_send(qp, &wr, &badwr) == 0, "post_send 20G");
    uint64_t charge20 = charge ? *charge : 0;
    uint64_t cbytes20 = charge_bytes ? *charge_bytes : 0;
    CHECK(poll_until(scq, 1, wc, 5000) == 1, "20G completion");
    uint64_t fast = now_ns() - t0;
    double cratio = charge20 ? (double)charge10 / (double)charge20 : 0.0;
    printf("bandwidth arm: charge 10G=%llu ns 20G=%llu ns ratio=%.2f "
           "(bytes %llu/%llu); wall 10G=%llu ns 20G=%llu ns\n",
           (unsigned long long)charge10, (unsigned long long)charge20, cratio,
           (unsigned long long)cbytes10, (unsigned long long)cbytes20,
           (unsigned long long)slow, (unsigned long long)fast);
    CHECK(cratio > 1.9 && cratio < 2.1,
          "2x line rate must halve the modelled charge, ratio %.2f", cratio);
    CHECK(cbytes10 == big && cbytes20 == big, "charge bytes %llu/%llu want %zu",
          (unsigned long long)cbytes10, (unsigned long long)cbytes20, big);
    CHECK(slow < fast * 4, "gross wall-clock sanity: 10G %llu ns vs 20G %llu ns",
          (unsigned long long)slow, (unsigned long long)fast);
    setenv("FAKEIB_LATENCY_US", "1.2", 1);
    setenv("FAKEIB_BW_GBPS", "400", 1);
    setenv("FAKEIB_LATENCY_US", "1.2", 1);
    setenv("FAKEIB_BW_GBPS", "400", 1);

    /* --- RDMA WRITE loopback moves the bytes --------------------------- */
    struct ibv_qp_init_attr qia2 = qia;
    qia2.recv_cq = rcq;   /* qp2's IMM completions land on the armed CQ */
    struct ibv_qp *qp2 = ibv_create_qp(pd, &qia2);
    CHECK(qp2 != NULL, "create_qp2");
    struct ibv_qp_attr q2;
    memset(&q2, 0, sizeof q2);
    q2.qp_state = IBV_QPS_INIT; q2.pkey_index = 0; q2.port_num = 1;
    CHECK(ibv_modify_qp(qp2, &q2, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT) == 0, "qp2 init");
    memset(&q2, 0, sizeof q2);
    q2.qp_state = IBV_QPS_RTR; q2.path_mtu = IBV_MTU_4096;
    q2.dest_qp_num = qp->qp_num; q2.rq_psn = 1;
    q2.ah_attr.is_global = 1; q2.ah_attr.port_num = 1;
    CHECK(ibv_modify_qp(qp2, &q2, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                  IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) == 0, "qp2 rtr");
    memset(&q2, 0, sizeof q2);
    q2.qp_state = IBV_QPS_RTS; q2.sq_psn = 1;
    CHECK(ibv_modify_qp(qp2, &q2, IBV_QP_STATE | IBV_QP_SQ_PSN) == 0, "qp2 rts");

    /* Retarget qp at qp2.  RTS->RTR is NOT a legal IB edge; real use (and
     * this test) goes RTS->RESET->INIT->RTR->RTS, which the shim must accept
     * and an impossible edge must still be refused. */
    qa.qp_state = IBV_QPS_RESET;
    CHECK(ibv_modify_qp(qp, &qa, IBV_QP_STATE) == 0, "qp -> RESET");
    memset(&qa, 0, sizeof qa);
    qa.qp_state = IBV_QPS_INIT; qa.pkey_index = 0; qa.port_num = 1;
    qa.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    CHECK(ibv_modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                IBV_QP_ACCESS_FLAGS) == 0, "qp reset->INIT");
    /* Send QP (qp) -> QP2 (qp2): point qp at qp2 and use qp2's recv CQ. */
    qa.qp_state = IBV_QPS_RTR;
    qa.dest_qp_num = qp2->qp_num;
    qa.rq_psn = 7;
    qa.ah_attr.is_global = 1;
    qa.ah_attr.grh.dgid.raw[15] = 0;
    CHECK(ibv_modify_qp(qp, &qa, IBV_QP_STATE | IBV_QP_AV | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) == 0,
          "qp -> qp2 RTR");
    qa.qp_state = IBV_QPS_RTS;
    CHECK(ibv_modify_qp(qp, &qa, IBV_QP_STATE) == 0, "qp -> qp2 RTS");

    char payload[4096];
    for (int i = 0; i < (int)sizeof payload; i++) payload[i] = (char)(i * 7 + 3);
    struct ibv_sge psge = {.addr = (uint64_t)(uintptr_t)payload, .length = sizeof payload, .lkey = mr->lkey};
    struct ibv_send_wr pw;
    memset(&pw, 0, sizeof pw);
    pw.wr_id = 0x4444;
    pw.opcode = IBV_WR_RDMA_WRITE;
    pw.send_flags = IBV_SEND_SIGNALED;
    pw.sg_list = &psge;
    pw.num_sge = 1;
    pw.wr.rdma.remote_addr = (uint64_t)(uintptr_t)rbuf + 128;
    pw.wr.rdma.rkey = rmr->rkey;
    CHECK(probe_post_send(qp, &pw, &badwr) == 0, "post RDMA_WRITE");
    CHECK(poll_until(scq, 1, wc, 1000) == 1, "RDMA_WRITE send completion");
    CHECK(wc[0].opcode == IBV_WC_RDMA_WRITE, "opcode %d", wc[0].opcode);
    CHECK(memcmp(rbuf + 128, payload, sizeof payload) == 0, "RDMA_WRITE landed the bytes");

    /* --- WRITE_WITH_IMM consumes a posted recv WR on the receiver ------ */
    struct ibv_recv_wr rwr;
    memset(&rwr, 0, sizeof rwr);
    rwr.wr_id = 0x5555;
    struct ibv_sge rsge = {.addr = (uint64_t)(uintptr_t)payload, .length = 0, .lkey = mr->lkey};
    rwr.sg_list = &rsge;
    rwr.num_sge = 1;
    CHECK(probe_post_recv(qp2, &rwr, &badrwr) == 0, "post_recv on qp2");
    pw.wr_id = 0x6666;
    pw.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    pw.imm_data = 0xdeadbeef;
    pw.wr.rdma.remote_addr = (uint64_t)(uintptr_t)rbuf;
    CHECK(probe_post_send(qp, &pw, &badwr) == 0, "post WRITE_WITH_IMM");
    CHECK(poll_until(scq, 1, wc, 1000) == 1, "WRITE_WITH_IMM send completion");
    struct ibv_cq *qcq2 = qia2.recv_cq;
    CHECK(poll_until(qcq2, 1, wc, 1000) == 1, "receiver IMM completion");
    CHECK(wc[0].wr_id == 0x5555, "recv wr_id 0x%llx", (unsigned long long)wc[0].wr_id);
    CHECK(wc[0].opcode == IBV_WC_RECV_RDMA_WITH_IMM, "recv opcode %d", wc[0].opcode);
    CHECK(wc[0].imm_data == 0xdeadbeef, "imm 0x%x", wc[0].imm_data);
    CHECK((wc[0].wc_flags & IBV_WC_WITH_IMM) != 0, "wc_flags WITH_IMM");
    CHECK(memcmp(rbuf, payload, sizeof payload) == 0, "WRITE_WITH_IMM landed the bytes");

    /* --- RDMA READ returns bytes into the local sge -------------------- */
    memset(payload, 0, sizeof payload);
    pw.wr_id = 0x7777;
    pw.opcode = IBV_WR_RDMA_READ;
    pw.wr.rdma.remote_addr = (uint64_t)(uintptr_t)rbuf;
    CHECK(probe_post_send(qp, &pw, &badwr) == 0, "post RDMA_READ");
    CHECK(poll_until(scq, 1, wc, 1000) == 1, "RDMA_READ completion");
    CHECK(wc[0].opcode == IBV_WC_RDMA_READ, "read opcode %d", wc[0].opcode);
    CHECK(wc[0].byte_len == sizeof payload, "read byte_len %u", wc[0].byte_len);
    CHECK(memcmp(payload, rbuf, sizeof payload) == 0, "RDMA_READ returned the bytes");

    /* --- comp-channel event fires ------------------------------------- */
    CHECK(ibv_req_notify_cq(rcq, 0) == 0, "req_notify_cq");
    /* The event fires on a RECEIVE completion, so qp2 needs a posted recv WR.
     * (The sender's WC goes to scq, which has no channel.) */
    rwr.wr_id = 0x8aaa;
    CHECK(probe_post_recv(qp2, &rwr, &badrwr) == 0, "post_recv for event");
    pw.wr_id = 0x8888;
    pw.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    pw.imm_data = 0x1234;
    CHECK(probe_post_send(qp, &pw, &badwr) == 0, "post for event");
    CHECK(poll_until(scq, 1, wc, 1000) == 1, "event send completion");
    uint64_t ev;
    ssize_t rd = read(chan->fd, &ev, sizeof ev);
    CHECK(rd == (ssize_t)sizeof ev && ev >= 1, "comp-channel event fired (rd=%zd)", rd);

    /* --- dmabuf (GDR) counter -----------------------------------------
     * A real registration carries a real fd (NCCL gets it from CUDA); the
     * shim treats fd < 0 as NCCL's capability PROBE (net_ib/gdr.cc:57) and
     * must not count it.  Both halves are asserted here, because counting the
     * probe made the first NCCL arm report a registration that never happened. */
    uint64_t *dmabuf_ctr = (uint64_t *)dlsym(RTLD_DEFAULT, "fakeib_dmabuf_registrations");
    CHECK(dmabuf_ctr != NULL, "dmabuf counter exported");
    uint64_t before = dmabuf_ctr ? *dmabuf_ctr : 0;
    struct ibv_mr *probe_mr = ibv_reg_dmabuf_mr(pd, 0, 0, 0, -1,
                                                IBV_ACCESS_LOCAL_WRITE);
    CHECK(probe_mr == NULL && errno != EOPNOTSUPP && errno != EPROTONOSUPPORT,
          "fd=-1 probe returns a non-fatal error (mr=%p errno=%d)",
          (void *)probe_mr, errno);
    CHECK(dmabuf_ctr && *dmabuf_ctr == before, "the probe did not count as a registration");
    int memfd = memfd_create("fakeib-dmabuf", 0);
    CHECK(memfd >= 0, "memfd_create");
    struct ibv_mr *dmr = ibv_reg_dmabuf_mr(pd, 0, 4096, 0x100000000ull, memfd,
                                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    CHECK(dmr != NULL, "reg_dmabuf_mr with a real fd");
    CHECK(dmabuf_ctr && *dmabuf_ctr == before + 1,
          "dmabuf registration counter moved exactly once (%llu -> %llu)",
          (unsigned long long)before, (unsigned long long)(dmabuf_ctr ? *dmabuf_ctr : 0));
    if (memfd >= 0) close(memfd);

    /* --- counters sanity ---------------------------------------------- */
    uint64_t *op = (uint64_t *)dlsym(RTLD_DEFAULT, "fakeib_post_send_calls");
    uint64_t *comp = (uint64_t *)dlsym(RTLD_DEFAULT, "fakeib_completions");
    CHECK(op && *op >= 6, "post_send counter %llu", op ? (unsigned long long)*op : 0);
    CHECK(comp && *comp >= 6, "completion counter %llu", comp ? (unsigned long long)*comp : 0);

    ibv_destroy_qp(qp2);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(rcq);
    ibv_destroy_cq(rcq1);
    ibv_destroy_cq(scq);
    ibv_destroy_comp_channel(chan);
    ibv_dereg_mr(dmr);
    ibv_dereg_mr(rmr);
    ibv_dereg_mr(mr);
    ibv_close_device(ctx);
    ibv_free_device_list(list);

    if (failures) { printf("PROBE FAILED: %d failure(s)\n", failures); return 1; }
    printf("PROBE OK\n");
    return 0;
}
