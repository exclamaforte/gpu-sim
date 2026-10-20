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

/* apply a link's bw/latency/jitter to nbytes (best-effort delay). */
int  fg_fab_transfer(fg_fab_t *f, int dir, size_t nbytes, double bw_gbps,
                     double latency_us, double jitter_us);

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