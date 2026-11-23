/* ece_shim.c -- ECE verbs for gpu-sim-7q3w.5's 2-rank NCCL arm.
 *
 * 7q3w.3's shim defined ibv_set_ece/ibv_query_ece as ENOSYS.  NCCL 2.32's
 * ncclIbSenderQpsCreate treats that as a QP setup failure
 * (transport/net_ib/connect.cc:788), so every allreduce died with
 * "unhandled system error" even though the bytes path was otherwise up.
 *
 * The modelled wire has no ECE negotiation: there is no vendor-specific
 * connection establishment to accept.  Returning success with a zeroed
 * ibv_ece (vendor_id 0, options 0, comp_mask 0) is the honest model of
 * "nothing negotiated", and it lets NCCL proceed to post the WRs the shim
 * actually carries.  That is asserted downstream by the fabric counters, not
 * assumed from this return value.
 *
 * Built alongside the 7q3w.3 shim with the original ECE definitions renamed:
 *     -Dibv_set_ece=fakeib_set_ece_orig
 *     -Dibv_query_ece=fakeib_query_ece_orig
 * so the version script exports these definitions instead.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <string.h>

/* The shim source is compiled with these names renamed so its ENOSYS
 * definitions do not collide with ours; keep our own definitions under the
 * real names despite those command-line macros. */
#undef ibv_set_ece
#undef ibv_query_ece

int ibv_set_ece(struct ibv_qp *qp, struct ibv_ece *ece) {
    (void)qp;
    if (!ece) return EINVAL;
    memset(ece, 0, sizeof *ece);
    return 0;
}

int ibv_query_ece(struct ibv_qp *qp, struct ibv_ece *ece) {
    (void)qp;
    if (!ece) return EINVAL;
    memset(ece, 0, sizeof *ece);
    return 0;
}
