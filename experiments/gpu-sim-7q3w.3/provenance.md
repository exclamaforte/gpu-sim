# gpu-sim-7q3w.3 — libibverbs.so.1 shim: provenance and state

Status: **phase 1 of the bead.** The shim's own semantics, the ABI pin and the
timing model are built and measured (`run.sh` stages 1–5, green). The NCCL arms
(device named by NCCL, IB transport chosen, all-reduce numerics, topology
injection) are **not** done yet: they need the synthetic sysfs and the
cross-process wire. Do not read this as acceptance for the bead.

## 1. Which header is compiled against, and why it is not the distro's

The ABI that crosses this shim is **NCCL's**, not rdma-core's newest. Two
headers were measured with `abi_layout.c` (the same program, one `-I` apart):

| struct | rdma-core v59.3 | NCCL v2.32.6 `ibvcore.h` |
|---|---|---|
| `ibv_send_wr` | 128 | 80 |
| `ibv_qp_init_attr` | 64 | 72 |
| `ibv_qp_attr` | 144 | 136 |

All other `sizeof`/`offsetof` values the shim touches are identical (the dumps
are in `abi_diff_expected.txt`, diffed by `run.sh` stage 2). The three
differences are trailing members added/removed between the two snapshots:
`_Static_assert`s in `ibverbs_shim.c` therefore pin **the NCCL sizes**, and the
shim compiles against `vendor/nccl/ibvcore.h`.

Vendored files and their recorded SHA256 (`vendor/SHA256SUMS`,
`vendor/nccl/SHA256SUMS`):

* `vendor/nccl/ibvcore.h` — NCCL tag `v2.32.6-1`
  (`src/include/ibvcore.h`), sha256 `3111a55c…c41f70d`. This is the header the
  wheel's `libnccl.so.2` was built against (NVIDIA keeps it in-tree for exactly
  the no-rdma-core build case).
* `vendor/infiniband/{verbs.h,verbs_api.h,ib_user_verbs.h,ib_user_ioctl_*.h}` —
  rdma-core **v59.3** (`libibverbs/verbs.h` byte-identical to the upstream tag,
  verified 2026-11-22). Kept as the measured counter-example, not as the build
  header.

A header swap that changes a field the shim touches moves the diff in stage 2
and fails the run; a wrong size that would corrupt NCCL's stack (the
`ibv_query_qp` `memset` case) fails the build instead.

## 2. The inline trap (why exporting symbols is not enough)

NCCL does **not** dlsym `ibv_post_send`, `ibv_post_recv` or `ibv_poll_cq`.
Those are static inlines in `src/include/ibvwrap.h` that dispatch through
`qp->context->ops.post_send`, `qp->context->ops.post_recv` and
`cq->context->ops.poll_cq`. `ibv_open_device` therefore fills the ops table
(`poll_cq` at byte 88, `post_send` at 200, `post_recv` at 208, asserted at
compile time). A shim that exports the symbols and leaves `ops` zeroed
segfaults on the first NCCL poll; `dlopen_check` and `abi_probe` both exercise
the ops path, not the exported names.

## 3. Timing model

Every completion is queued with `ready_ns = now + bytes/(bandwidth) + latency`
and `poll_cq` returns it only once `CLOCK_MONOTONIC` has reached `ready_ns`. No
sleeps in the shim. Defaults model one NDR port: **400 Gb/s**, **1.2 µs**. Two
env overrides exist so the acceptance's 2× assertion drives the same code path a
spec file will: `FAKEIB_BW_GBPS`, `FAKEIB_LATENCY_US`.

Measured (stage 5, this box):

* 40 ms latency arm: immediate poll returns 0, first completion at
  ≈40.0 ms (the check is on the wire, not on the wall clock of the test).
* 2× line rate on 1 MiB, asserted on the shim's own modelled charge
  (`fakeib_last_charge_ns`): **ratio 2.00** (band [1.9, 2.1]).  The same arm
  prints the wall-clock times but does not assert on them: on a loaded box one
  scheduler slice made the WALL ratio 32.76 while the charge ratio was exactly
  2.00.  A wall-clock ratio measures the box; the charge ratio measures the
  model.

## 4. Counters

Exported and also appended as JSON to `$FAKEIB_COUNTERS_FILE` at exit:
`fakeib_dmabuf_registrations` (GDR arm), `reg_mr_calls`, `post_send_calls`,
`post_recv_calls`, `completions`, `bytes_written`, `bytes_read`, `writes`,
`reads`, `sends`, `recvs`, `qp_transition_refusals`, `rnr_drops`,
`remote_access_errors`, `cq_full_drops`. The acceptance's "dmabuf counter
non-zero" item is the first of these; `abi_probe` already asserts it.

Measured now: the shim loads by SONAME like NCCL loads it; the 38-name dlsym
surface is complete; the QP state machine accepts RESET→INIT→RTR→RTS and refuses
RTS→INIT; one posted WR gives exactly one completion with the right `wr_id`,
opcode and `byte_len`; an empty CQ returns 0; RDMA WRITE, WRITE_WITH_IMM
(including the receiver's IMM completion) and READ move the bytes; the
comp-channel event fires; `ibv_reg_dmabuf_mr` increments its counter; the wire
model gates completions and scales with line rate.

Also measured, cross-process (`wire_probe`, two real processes): the two ranks
exchange (qpn,rkey,addr) over pipes exactly as NCCL does and drive RDMA WRITE,
WRITE_WITH_IMM (receiver IMM completion + imm data) and READ across the
MAP_SHARED ring; the sender's completion for a 20 ms latency arm lands at
≥20 ms measured on its own clock.

## 6. The cross-process ring, and its model's limits

One byte-ring file per (from,to) rank pair under `FAKEIB_RING_DIR` (default
`.lab/tmp/7q3w3-rings`), `FAKEIB_RING_MB` = 64 MiB per direction, records
8-aligned with a `len/kind/wr_id/addr/rkey/imm/dest_qpn/ready_ns/nbytes/status/tag`
header.  `FAKEIB_NRANKS` (default 2) tells each rank which pair-rings to open in
both directions at startup, because a receiver cannot see an inbound ring until
it opens the file.  A full ring is a 30 s stall then a counted WR failure, never
a silent drop.  `rkey` is resolved on the RECEIVING rank in its own MR table (so
both processes may own rkey 1); `remote_addr - iova` is the offset inside that
MR.

Deliberate model limits, stated so nobody mistakes them for hardware: an
invalid rkey/range is counted on the RECEIVER (`remote_access_errors`) and does
not come back as the sender's WC status; a SEND completes at post time (the
ring carries it) instead of waiting for the peer's posted recv WR; the bytes are
in the ring immediately and only the completion is delayed by the model.  The
NCCL data path uses RDMA WRITE / WRITE_WITH_IMM / READ only (net_ib's p2p.cc),
so the first two limits do not sit on the collective path.

## 7. What is still missing

Missing (next commits): the synthetic sysfs (a per-rank mount namespace with a
fake /sys/class/infiniband, so NCCL builds two nodes with an IB NET edge) and
the NCCL acceptance arms (a)–(f) from the bead.  The wire itself is built and
measured (section 6).

## 8. Reproduce

```bash
cd .lab/wt/gpu-sim-7q3w.3
CUDA_VISIBLE_DEVICES= experiments/gpu-sim-7q3w.3/run.sh
```

CPU only, no GPU queue needed, scratch under `.lab/tmp/` (project-local).
