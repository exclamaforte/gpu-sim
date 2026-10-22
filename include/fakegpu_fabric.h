#ifndef FAKEGPU_FABRIC_H
#define FAKEGPU_FABRIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FG_FAB_MAGIC 0x47505553494d4642ULL

#define FG_FAB_OK            0
#define FG_FAB_ETIMEDOUT     1
#define FG_FAB_EINVAL        2
#define FG_FAB_ESIZE         3
#define FG_FAB_EPEERDEAD     4
#define FG_FAB_ESTALE        5
#define FG_FAB_ENOMEM        6
#define FG_FAB_EIO           8
#define FG_FAB_ELOSS         9
#define FG_FAB_EBUSY         7

#define FG_FAB_MAX_PAYLOAD   (1 << 20)   /* 1 MiB */

typedef struct fg_fab fg_fab_t;

/* Attach to (or create) the named segment for run_id.
 * epoch: caller's epoch; a pre-existing segment with a different epoch is
 *        rejected with FG_FAB_ESTALE instead of being reused.
 * timeout_ns: deadline for message/barrier waits inside this handle
 *             (0 -> 1s default). The attach wait itself uses a separate,
 *             longer deadline (GPUSIM_ATTACH_TIMEOUT_NS, default 5s, never
 *             below timeout_ns) because the creator must materialize the
 *             whole segment before state==1 becomes visible.
 * Returns NULL on failure and sets *err. */
fg_fab_t *fg_fab_attach(const char *run_id, uint64_t epoch, int rank, int nranks,
                        uint64_t timeout_ns, int *err);

int  fg_fab_barrier(fg_fab_t *f);

/* send exactly n bytes (n <= FG_FAB_MAX_PAYLOAD) to peer. */
int  fg_fab_send(fg_fab_t *f, int peer, const void *buf, size_t n);

/* receive exactly n bytes from `from`; n must match the sent length. */
int  fg_fab_recv(fg_fab_t *f, int from, void *buf, size_t n);

/* receive up to cap bytes; accepts any message with len <= cap and reports
   actual bytes received in *got. */
int  fg_fab_recv_max(fg_fab_t *f, int from, void *buf, size_t cap, size_t *got);

/* Decider variants: same contracts, plus (slot, seq) of the fabric slot that
   carried the payload and the sender's global consumed_seq witness. */
int  fg_fab_send_x(fg_fab_t *f, int peer, const void *buf, size_t n,
                   int *slot_out, unsigned *seq_out);
int  fg_fab_send_reg(fg_fab_t *f, int peer, size_t reg_offset, size_t n,
                     int *slot_out, unsigned *seq_out);
size_t fg_fab_reg_mr(fg_fab_t *f, const void *buf, size_t size);
int  fg_fab_recv_max_ex(fg_fab_t *f, int from, void *buf, size_t cap, size_t *got,
                        int *slot_out, unsigned *seq_out, unsigned *consumed_out);

/* Tag-aware paths (gpu-sim-ast).  A request only ever consumes a message that
   carries its own tag; want_tag < 0 means "any tag" (legacy behaviour).  NCCL
   distinguishes concurrent message streams by tag, so delivering whichever
   message happens to be next hands a piece to a request that belongs to a
   different stream. */
int  fg_fab_recv_tag_ex(fg_fab_t *f, int from, void *buf, size_t cap, int want_tag,
                        size_t *got, int *slot_out, unsigned *seq_out, unsigned *consumed_out);
int  fg_fab_send_tag(fg_fab_t *f, int peer, const void *buf, size_t n, int msg_tag);
int  fg_fab_send_id(fg_fab_t *f, int peer, const void *buf, size_t n, int msg_tag, int msg_chan);
int  fg_fab_recv_id_ex(fg_fab_t *f, int from, void *buf, size_t cap, int want_tag, int want_chan,
                       size_t *got, int *slot_out, unsigned *seq_out, unsigned *consumed_out);

/* query fault-injection counters (for invariant tests) */
uint64_t fg_fab_success_count(fg_fab_t *f);
uint64_t fg_fab_attempt_count(fg_fab_t *f);

/* apply a link's bw/latency/jitter/loss/partition/timeout to nbytes.
   loss_ppm: per-transfer drop probability (ppm). partition_after: number of
   successful transfers before the link partitions (0 = never). timeout_ns:
   hard deadline. timeout_us: per-transfer deadline (0 = none). Returns
   FG_FAB_OK on success, FG_FAB_ELOSS on loss, FG_FAB_ETIMEDOUT on
   partition/timeout, FG_FAB_EPEERDEAD on peer death. */
int  fg_fab_transfer(fg_fab_t *f, int dir, size_t nbytes, double bw_gbps,
                     double latency_us, double jitter_us, int loss_ppm,
                     uint64_t partition_after, double timeout_us, int link_id);

void fg_fab_detach(fg_fab_t *f);

/* remove the segment (call after all ranks detached). */
void fg_fab_unlink(const char *run_id);

/* introspection */
int  fg_fab_nranks(const fg_fab_t *f);
int  fg_fab_rank(const fg_fab_t *f);
const char *fg_fab_errstr(int err);

#ifdef __cplusplus
}
#endif

#endif /* FAKEGPU_FABRIC_H */