/* gpu-sim-7q3w.3: dump the libibverbs ABI layout this header defines.
 *
 * Build TWICE -- once against the vendored rdma-core v59.3 header we compile
 * the shim with, once against the header NCCL itself was built against -- and
 * diff the output.  A silently different offset in ctx->ops is the 7q3w
 * SIGSEGV class, so the experiment's run.sh diffs these two dumps before any
 * NCCL run is allowed to start.
 */
#include <stddef.h>
#include <stdio.h>
#include IBV_HEADER

#define P(s) printf("%-40s %zu\n", "sizeof(" #s ")", sizeof(struct s))
#define O(s, f) printf("%-40s %zu\n", "offsetof(" #s "," #f ")", offsetof(struct s, f))

int main(void) {
    P(ibv_device);
    P(ibv_context);
    O(ibv_context, device);
    O(ibv_context, ops);
    P(ibv_context_ops);
    O(ibv_context_ops, poll_cq);
    O(ibv_context_ops, req_notify_cq);
    O(ibv_context_ops, post_send);
    O(ibv_context_ops, post_recv);
    P(ibv_pd);
    P(ibv_mr);
    O(ibv_mr, addr);
    O(ibv_mr, rkey);
    P(ibv_cq);
    O(ibv_cq, context);
    O(ibv_cq, channel);
    P(ibv_qp);
    O(ibv_qp, context);
    O(ibv_qp, qp_context);
    O(ibv_qp, state);
    O(ibv_qp, qp_num);
    P(ibv_send_wr);
    O(ibv_send_wr, wr_id);
    O(ibv_send_wr, next);
    O(ibv_send_wr, sg_list);
    O(ibv_send_wr, num_sge);
    O(ibv_send_wr, opcode);
    O(ibv_send_wr, send_flags);
    O(ibv_send_wr, wr.rdma.remote_addr);
    O(ibv_send_wr, wr.rdma.rkey);
    P(ibv_recv_wr);
    O(ibv_recv_wr, wr_id);
    O(ibv_recv_wr, next);
    O(ibv_recv_wr, sg_list);
    O(ibv_recv_wr, num_sge);
    P(ibv_sge);
    O(ibv_sge, addr);
    O(ibv_sge, length);
    O(ibv_sge, lkey);
    P(ibv_wc);
    O(ibv_wc, wr_id);
    O(ibv_wc, status);
    O(ibv_wc, opcode);
    O(ibv_wc, vendor_err);
    O(ibv_wc, byte_len);
    O(ibv_wc, qp_num);
    O(ibv_wc, src_qp);
    O(ibv_wc, wc_flags);
    O(ibv_wc, pkey_index);
    O(ibv_wc, slid);
    O(ibv_wc, sl);
    O(ibv_wc, dlid_path_bits);
    P(ibv_device_attr);
    O(ibv_device_attr, max_qp_wr);
    O(ibv_device_attr, max_sge);
    O(ibv_device_attr, max_sge_rd);
    O(ibv_device_attr, max_cqe);
    O(ibv_device_attr, max_mr);
    P(ibv_port_attr);
    O(ibv_port_attr, state);
    O(ibv_port_attr, max_mtu);
    O(ibv_port_attr, active_mtu);
    O(ibv_port_attr, gid_tbl_len);
    O(ibv_port_attr, active_speed);
    O(ibv_port_attr, active_width);
    P(ibv_qp_init_attr);
    O(ibv_qp_init_attr, qp_context);
    O(ibv_qp_init_attr, send_cq);
    O(ibv_qp_init_attr, recv_cq);
    O(ibv_qp_init_attr, srq);
    O(ibv_qp_init_attr, cap);
    O(ibv_qp_init_attr, qp_type);
    P(ibv_qp_attr);
    O(ibv_qp_attr, qp_state);
    O(ibv_qp_attr, path_mtu);
    O(ibv_qp_attr, dest_qp_num);
    O(ibv_qp_attr, rq_psn);
    O(ibv_qp_attr, sq_psn);
    O(ibv_qp_attr, max_rd_atomic);
    O(ibv_qp_attr, max_dest_rd_atomic);
    O(ibv_qp_attr, min_rnr_timer);
    O(ibv_qp_attr, timeout);
    O(ibv_qp_attr, retry_cnt);
    O(ibv_qp_attr, rnr_retry);
    O(ibv_qp_attr, ah_attr);
    P(ibv_ah);
    O(ibv_ah, context);
    O(ibv_ah, pd);
    P(ibv_comp_channel);
    O(ibv_comp_channel, fd);
    return 0;
}
