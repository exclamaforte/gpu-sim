/* gpu-sim-7q3w.3: the libibverbs surface NCCL 2.32.6 dlsyms, declared here.
 *
 * NCCL's ibvcore.h intentionally carries structs/enums only (it is the ABI
 * snapshot, not the API header), so the shim and its probes need these
 * declarations.  Signatures are NCCL v2.32.6-1 src/include/ibvwrap.h's.
 *
 * The post/recv/poll entry points are NOT in this list: NCCL reaches them
 * through qp->context->ops.{post_send,post_recv} and cq->context->ops.poll_cq
 * (static inlines in its ibvwrap.h).  probe_poll_cq/probe_post_recv below use
 * exactly that path so the probe fails if the ops table is ever left zeroed.
 */
#ifndef FAKEIB_API_H_
#define FAKEIB_API_H_

#include "nccl/ibvcore.h"

struct ibv_device **ibv_get_device_list(int *num);
void ibv_free_device_list(struct ibv_device **list);
const char *ibv_get_device_name(struct ibv_device *dev);
int ibv_get_device_index(struct ibv_device *dev);
struct ibv_context *ibv_open_device(struct ibv_device *dev);
int ibv_close_device(struct ibv_context *context);
int ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr);
int ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr);
int ibv_query_port_speed(struct ibv_context *context, uint8_t port_num, uint64_t *speed);
int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
int ibv_query_pkey(struct ibv_context *context, uint8_t port_num, int index, uint16_t *pkey);
int ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask,
                 struct ibv_qp_init_attr *init_attr);
struct ibv_pd *ibv_alloc_pd(struct ibv_context *context);
int ibv_dealloc_pd(struct ibv_pd *pd);
struct ibv_mr *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access);
struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t length,
                                uint64_t iova, unsigned int access);
struct ibv_mr *ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length,
                                 uint64_t iova, int fd, int access);
int ibv_dereg_mr(struct ibv_mr *mr);
struct ibv_comp_channel *ibv_create_comp_channel(struct ibv_context *context);
int ibv_destroy_comp_channel(struct ibv_comp_channel *channel);
struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe, void *cq_context,
                             struct ibv_comp_channel *channel, int comp_vector);
int ibv_destroy_cq(struct ibv_cq *cq);
struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
int ibv_destroy_qp(struct ibv_qp *qp);
struct ibv_ah *ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr);
int ibv_destroy_ah(struct ibv_ah *ah);
struct ibv_srq *ibv_create_srq(struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr);
int ibv_destroy_srq(struct ibv_srq *srq);
int ibv_fork_init(void);
int ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event);
void ibv_ack_async_event(struct ibv_async_event *event);
const char *ibv_event_type_str(enum ibv_event_type event);
int ibv_set_ece(struct ibv_qp *qp, struct ibv_ece *ece);
int ibv_query_ece(struct ibv_qp *qp, struct ibv_ece *ece);
int ibv_req_notify_cq(struct ibv_cq *cq, int solicited_only);
int ibv_get_cq_event(struct ibv_comp_channel *channel, struct ibv_cq **cq, void **cq_context);
void ibv_ack_cq_events(struct ibv_cq *cq, unsigned int nevents);

static inline int probe_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc) {
    return cq->context->ops.poll_cq(cq, num_entries, wc);
}
static inline int probe_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                                  struct ibv_recv_wr **bad_wr) {
    return qp->context->ops.post_recv(qp, wr, bad_wr);
}
static inline int probe_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                                  struct ibv_send_wr **bad_wr) {
    return qp->context->ops.post_send(qp, wr, bad_wr);
}
/* NCCL's own 32-name dlsym list (measured from libnccl.so.2, see the bead). */
#define FAKEIB_DLSYM_SURFACE(X)                                            \
    X(ibv_get_device_list) X(ibv_free_device_list) X(ibv_get_device_name)   \
    X(ibv_open_device) X(ibv_close_device) X(ibv_query_device)              \
    X(ibv_query_port) X(ibv_query_port_speed) X(ibv_query_gid)              \
    X(ibv_query_pkey) X(ibv_query_qp) X(ibv_alloc_pd) X(ibv_dealloc_pd)     \
    X(ibv_reg_mr) X(ibv_reg_mr_iova2) X(ibv_reg_dmabuf_mr) X(ibv_dereg_mr)  \
    X(ibv_create_comp_channel) X(ibv_destroy_comp_channel)                  \
    X(ibv_create_cq) X(ibv_destroy_cq) X(ibv_create_qp) X(ibv_destroy_qp)   \
    X(ibv_modify_qp) X(ibv_create_ah) X(ibv_destroy_ah) X(ibv_create_srq)   \
    X(ibv_destroy_srq) X(ibv_fork_init) X(ibv_get_async_event)              \
    X(ibv_ack_async_event) X(ibv_event_type_str) X(ibv_set_ece)             \
    X(ibv_query_ece)

#endif
