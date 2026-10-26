"""Run one training step under a parallelism configuration, and emit a trace.

This is the worker process: torchrun starts RANK of WORLD_SIZE, each one owns one
device (the real GPU on this box), and every collective goes through the
interposer named in LD_PRELOAD. The training loop is ordinary torch -- there is no
simulation in this file. Everything that is faked is faked below the Python line,
which is the property the project exists to preserve: the same program runs
against real hardware and against the model without being rewritten.

What the harness adds on top of plain torch:

  --steps N            number of steps to time (default 5)
  --delay-ms LO HI     synthetic per-microbatch compute delay, uniform, drawn
                       independently per step. This is the "random kernel delay"
                       mode: it stands in for compute timing that has no
                       calibration yet, so the trace has plausible structure
                       while the COLLECTIVE side is what we actually compare.
  --json OUT           write a per-rank summary (step times, loss, bytes moved)

The trace itself comes from the interposer (GPUSIM_TRACE), not from here: this
process only records what it can see at the Python level, and the two are joined
by rank and timestamp.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist

from tools.torch_parallel.model import ModelSpec, PPStage, TinyLM, TinyLMTP


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5)
    # The first step pays CUDA context + cuBLAS init + kernel compilation:
    # measured 556 ms against 11 ms for steady state. Averaging it in makes a
    # sweep compare initialization cost rather than the parallel config, which
    # is exactly the confound this harness exists to avoid.
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--delay-ms", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--json", type=str, default=None)
    # Capture REAL kernels and Python frames with torch.profiler. The interposer
    # only knows about collectives; everything above it -- the Python stack, aten
    # ops, CUDA kernels -- is invisible to it and has to come from the profiler.
    p.add_argument("--torch-profile", type=str, default=None,
                   help="export a torch.profiler chrome trace to <prefix>.rankN.json")
    # Parallelism decomposition. world_size == pp * dp * tp is enforced; the
    # rank layout is rank = pp_idx*(tp*dp) + dp_idx*tp + tp_idx, so each
    # pipeline stage is a contiguous block of tp*dp ranks and TP groups stay
    # contiguous slices inside a stage (the pp==1 layout is unchanged).
    p.add_argument("--tp", type=int, default=1,
                   help="tensor-parallel width; world_size/tp is the data-parallel width")
    p.add_argument("--pp", type=int, default=1,
                   help="pipeline-parallel width: n_layers splits across pp stages")
    p.add_argument("--cp", type=int, default=1,
                   help="context-parallel width: sequence splits across cp ranks (ring attention)")
    p.add_argument("--micro-batches", type=int, default=2,
                   help="GPipe micro-batches per step (must divide --batch)")
    # ZeRO stage: 1 = plain DP, 2 = shard grads + optimizer states, 3 = shard
    # parameters too (via FSDP). Verified on the dp axis only (pp==1, tp==1);
    # tp/pp combinations are refused explicitly until verified, not silently run.
    p.add_argument("--zero", type=int, default=1, choices=[1, 2, 3],
                   help="ZeRO stage; 2/3 need dp>1 and pp==tp==1")
    # gpu-sim-f50 ZeRO-2 gate: --save-flat (world1) writes the concatenated
    # full-batch grads; --verify-flat (world2 zero2) all_gathers the RS shards
    # back and asserts reassembled/dp == single-process grads. Reference is a
    # real single-process run, never a traced value.
    p.add_argument("--save-flat", type=str, default=None,
                   help="path to save concatenated grads (dp==1 reference)")
    p.add_argument("--verify-flat", type=str, default=None,
                   help="path to single-process flat grads to verify against")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--cp-grad-sync", type=str, default="reducescatter", choices=["allreduce", "reducescatter"],
                   help="cp gradient sync mode: allreduce (crude) or reducescatter (equivalence-preserving; same byte cost as allreduce in this harness because all_gather reconstructs full grad; true production reduce-scatter needs sharded optimizer)")
    p.add_argument("--optim", type=str, default="adamw", choices=["adamw", "sgd"],
                   help="optimizer; sgd exposes gradient-scale bugs AdamW's adaptivity hides")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="learning rate for --optim")
    # gpu-sim-tbp convention (real DDP): disjoint rank shards of the batch +
    # explicit 1/dp averaging of the summed gradients. Same-batch-no-scale
    # would leave world-dp grads dp x too big, and AdamW would nearly hide
    # it -- that false pass is exactly what this flag exists to prevent.
    # Required whenever the transport is on (dp>1, SIMULATE_ONLY=0); refused
    # loudly otherwise so no silent wrong-convention run is possible.
    p.add_argument("--dp-mean", action="store_true",
                   help="rank-shard the batch and average summed grads by 1/dp")
    return p.parse_args()


def _zero_mem_model(param_bytes: float, dp: int, zero: int) -> dict:
    """Analytic per-rank sharded-STATE model (bytes, no activations).

    P = full params; grads = P; AdamW states = 2P. ZeRO-1 shards states,
    ZeRO-2 shards grads too, ZeRO-3 (FSDP) shards all three. Activations are
    excluded: no ZeRO stage shards them, so they are the floor the measured
    peak cannot go below. Compare against measured peak_alloc_MB; the gap is
    the finding, not an embarrassment."""
    P = param_bytes
    if zero == 1:
        parts = {"params": P, "grads": P, "opt_states": 2 * P / dp}
    elif zero == 2:
        parts = {"params": P, "grads": P / dp, "opt_states": 2 * P / dp}
    else:
        parts = {"params": P / dp, "grads": P / dp, "opt_states": 2 * P / dp}
    return {k: round(v / 1e6, 1) for k, v in parts.items()} | \
        {"state_total_MB": round(sum(parts.values()) / 1e6, 1)}


def tp_groups(pp, tp, cp, dp):
    """TP groups per the PI decomposition: rank = ((pp_idx*tp + tp_rank)*cp + cp_rank)*dp + dp_rank.
    A tp group fixes (pp_idx, cp_rank, dp_rank) and varies tp_rank."""
    groups = []
    for p in range(pp):
        for c in range(cp):
            for d in range(dp):
                ranks = [((p * tp + t) * cp + c) * dp + d for t in range(tp)]
                groups.append(ranks)
    return groups


def pp_neighbors(rank, pp, tp, cp, dp):
    """Previous/next PP stage ranks, keeping tp_rank, cp_rank, dp_rank fixed."""
    pp_idx = rank // (tp * cp * dp)
    sub = rank % (tp * cp * dp)
    tp_rank = sub // (cp * dp)
    sub2 = sub % (cp * dp)
    cp_rank = sub2 // dp
    dp_rank = sub2 % dp
    prev_rank = None
    next_rank = None
    if pp_idx > 0:
        prev_rank = (((pp_idx - 1) * tp + tp_rank) * cp + cp_rank) * dp + dp_rank
    if pp_idx < pp - 1:
        next_rank = (((pp_idx + 1) * tp + tp_rank) * cp + cp_rank) * dp + dp_rank
    return prev_rank, next_rank


def main() -> int:
    args = _parse()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # rank decomposition first (needed for seeding)
    tp = args.tp
    pp = args.pp
    if world % (pp * tp * args.cp) != 0:
        raise SystemExit(
            f"world_size {world} is not divisible by pp*tp*cp {pp * tp * args.cp}")
    cp = args.cp
    dp = world // (pp * tp * cp)
    pp_idx = rank // (tp * cp * dp)
    sub = rank % (tp * cp * dp)
    tp_rank = sub // (cp * dp)
    sub2 = sub % (cp * dp)
    cp_rank = sub2 // dp
    dp_rank = sub2 % dp

    # gpu-sim-tbp: --dp-mean needs IDENTICAL inits on every rank.
    # gpu-sim-7sb: seed by replica identity (pp_idx, tp_rank), NOT world rank,
    # so cp ranks (replicas) share weights while dp ranks differ.
    if args.dp_mean:
        init_seed = args.seed
    else:
        init_seed = args.seed + (pp_idx * tp + tp_rank) * 1000 + dp_rank
    torch.manual_seed(init_seed)
    random.seed(init_seed)

    if world > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    # All ranks share the one REAL GPU: the multi-device fan-out is simulated by
    # the interposer, not by hardware. Asking for device==local_rank is what
    # "invalid device ordinal" means here -- there is no second physical card.
    device_index = int(os.environ.get("HARNESS_DEVICE", "0"))
    torch.cuda.set_device(device_index)

    zero = args.zero
    if cp > 1 and pp > 1:
        raise SystemExit("--cp>1 with --pp>1 not verified in v1")
    if cp > 1 and zero > 1:
        raise SystemExit("--cp>1 with --zero>1 not verified in v1")
    if zero > 1:
        # Verified on the dp axis only. FSDP over a TP-sharded model or a PP
        # stage is plausible but unverified; refusing beats silently running.
        if pp != 1 or tp != 1:
            raise SystemExit(
                f"--zero {zero} needs pp==tp==1 (dp axis only), "
                f"got pp={pp} tp={tp}")
        if dp < 2:
            raise SystemExit(f"--zero {zero} needs dp>1 (world {world})")

    _n_kv = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
    spec = ModelSpec(vocab=args.vocab, seq=args.seq, d_model=args.d_model,
                     n_layers=args.layers, n_heads=args.n_heads,
                     n_kv_heads=_n_kv, d_ff=args.d_model * 4)
    tp_group = dp_group = None
    cp_group = None
    if dp > 1:
        for p in range(pp):
            for t in range(tp):
                for c in range(cp):
                    ranks = [((p * tp + t) * cp + c) * dp + d for d in range(dp)]
                    g = dist.new_group(ranks)
                    if rank in ranks:
                        dp_group = g
    if cp > 1:
        for p in range(pp):
            for t in range(tp):
                for d in range(dp):
                    ranks = [((p * tp + t) * cp + c) * dp + d for c in range(cp)]
                    g = dist.new_group(ranks)
                    if rank in ranks:
                        cp_group = g
    if tp > 1:
        for ranks in tp_groups(pp, tp, cp, dp):
            g = dist.new_group(ranks)
            if rank in ranks:
                tp_group = g
        if pp == 1:
            # Deterministic sharding: every rank draws the full weights and
            # slices its tp shard, so sharded and reference agree (f50 gate).
            cp_peers = [((pp_idx * tp + tp_rank) * cp + c) * dp + dp_rank for c in range(cp)] if cp > 1 else [rank]
            model = TinyLMTP(spec, tp_size=tp, group=tp_group,
                             tp_rank=tp_rank, weight_seed=args.seed,
                             cp_size=cp, cp_rank=cp_rank,
                             cp_peers=cp_peers, cp_group=cp_group).cuda()
        else:
            model = PPStage(spec, pp_idx, pp,
                            tp_size=tp, group=tp_group, tp_rank=tp_rank,
                            weight_seed=args.seed).cuda()
    elif pp > 1:
        model = PPStage(spec, pp_idx, pp, weight_seed=args.seed).cuda()
    else:
        cp_peers = [((pp_idx * tp + tp_rank) * cp + c) * dp + dp_rank for c in range(cp)] if cp > 1 else [rank]
        model = TinyLM(spec, cp_size=cp, cp_rank=cp_rank,
                       cp_peers=cp_peers, cp_group=cp_group).cuda()
        if zero == 3:
            # ZeRO-3 = fully sharded params (FSDP). Verified viable under the
            # timing-only interposer: the all_gathered payload is garbage
            # (uninitialized device memory) but FSDP runs to completion and
            # emits real allgather/reducescatter traffic. Same bargain as the
            # rest of the harness: traffic and timing are real, values are not.
            # (Needed a one-line interposer fix first: ncclReduceScatter took
            # recvcount as a pointer; real NCCL passes size_t by value. The
            # wrong signature segfaulted FSDP's backward -- fixed in this bead
            # with tests/test_nccl_rs_signature.py pinning it.)
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            model = FSDP(model, process_group=dp_group,
                         device_id=torch.device("cuda", device_index))
    # NOT torch.nn.parallel.DistributedDataParallel.
    #
    # DDP cross-checks parameter counts with a real all_gather, so it needs the
    # collective to MOVE DATA. The interposer simulates timing only (everything
    # else would require a transport, which is the thing it replaces), so DDP
    # aborts with "rank 0 has inconsistent 0 params". Data-parallel gradient
    # synchronisation is therefore done explicitly below -- which is exactly what
    # DDP does internally, minus the bookkeeping that depends on payload.
    #
    # Consequence, stated so nobody is surprised: after all_reduce the gradient
    # is NOT the true sum across ranks, and after send/recv the receiver keeps
    # its init value (measured: sender filled 8x8 with 1.0, receiver still read
    # 2.0 -- the payload never moves). Timing and traffic are real (every
    # about model convergence. The loss printed in PP mode is garbage.
    # gpu-sim-tbp exception: with --dp-mean AND the transport on
    # (FAKEGPU_NCCL_SIMULATE_ONLY=0), the DP all_reduce really sums and the
    # 1/dp averaging restores the exact mean -- that path is data-correct
    # by construction and proven by the tbp gate, not by this paragraph.
    # gpu-sim-tbp: the optimizer is a choice because AdamW's adaptivity hides
    # gradient-scale bugs (a 2x grad barely moves an AdamW trajectory, but
    # exactly doubles an SGD step). The equivalence proof runs SGD.
    if args.optim == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # One batch shape for every step: the comparison across parallelism configs
    # must not be confounded by a changing workload.
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randint(0, spec.vocab, (args.batch, spec.seq + 1),
                      device="cuda", generator=g)
    # gpu-sim-tbp convention: with --dp-mean each rank trains on a DISJOINT
    # shard of the batch (drawn identically everywhere, then sliced by the
    # dp-group rank), and the summed gradients are averaged by 1/dp below.
    # Shard + average reproduces the exact single-process full-batch math;
    # same-batch-no-scale would double the update and AdamW would hide it.
    dp_mean = bool(args.dp_mean)
    if dp_mean and (tp > 1 or pp > 1):
        raise SystemExit("--dp-mean is verified for the dp axis only "
                         "(tp==pp==1)")
    # gpu-sim-f50: zero 2/3 run the same dp loop with real reduce_scatter /
    # FSDP collectives; the shard+average convention covers them too.
    if dp > 1 and not dp_mean and zero == 1 and \
            os.environ.get("FAKEGPU_NCCL_SIMULATE_ONLY") == "0":
        raise SystemExit("transport is on with dp>1 but --dp-mean is off: "
                         "summed same-batch gradients would be dp x too big. "
                         "Pass --dp-mean.")
    if dp_mean:
        if args.batch % dp != 0:
            raise SystemExit(f"--dp-mean needs batch divisible by dp "
                             f"({args.batch} % {dp})")
        # dp_group is None when dp==1: the shard is the whole batch.
        dp_rank = dist.get_rank(group=dp_group) \
            if (world > 1 and dp_group is not None) else 0
        bsz = args.batch // dp
        x = x[dp_rank * bsz:(dp_rank + 1) * bsz].contiguous()

    step_times: list[float] = []
    measure_start_ns: int | None = None
    losses: list[float] = []
    n_warm = max(0, min(args.warmup, args.steps - 1))
    use_pp = pp > 1
    if use_pp:
        if args.batch % args.micro_batches != 0:
            raise SystemExit(
                f"batch {args.batch} not divisible by "
                f"micro-batches {args.micro_batches}")
        mb = args.batch // args.micro_batches
        mbs = args.micro_batches
        prev_rank, next_rank = pp_neighbors(rank, pp, tp, cp, dp)
        act_shape = (mb, spec.seq, spec.d_model)
    # Clock-domain anchors. The interposer stamps collectives with CLOCK_REALTIME
    # (epoch ns); torch.profiler stamps kernels with CLOCK_MONOTONIC (boot us).
    # Aligning them after the fact means guessing, and a wrong offset silently
    # slides every collective off the kernels it explains. Capture both here, at
    # the same instant, so the merger can convert exactly.
    clock_anchor = {"mono_us": time.monotonic() * 1e6, "real_ns": time.time_ns()}
    # Kineto's timestamps live in their own base -- measured: torch ts 1.97e12 vs
    # time.monotonic() 4.06e11 in the same run, a 1.568e12 us gap. So neither
    # clock can be used directly to place collectives on the kernel timeline.
    # Emitting an anchor EVENT lets the merger align on a timestamp both traces
    # contain, with no assumptions about either clock's base.
    anchor_scope = None

    prof = None
    if args.torch_profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_stack=True,          # Python frames: this is the ask
            record_shapes=False,
            profile_memory=False,
        )
        prof.__enter__()
        # Captured HERE, not before __enter__: CUPTI initialisation takes seconds
        # (measured ~7 s), so an anchor taken before it would place every
        # collective seconds away from the kernels it issued.
        clock_anchor = {"mono_us": time.monotonic() * 1e6, "real_ns": time.time_ns()}
        anchor_scope = torch.profiler.record_function("fg_clock_anchor")
        anchor_scope.__enter__()
        anchor_scope.__exit__(None, None, None)  # zero-length marker at the anchor

    for step in range(args.steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if not use_pp:
            with torch.profiler.record_function("forward"):
             if zero == 3:
                # FSDP does not expose the inner module's loss; call forward
                # and score outside, same math as TinyLM.loss.
                logits = model(x[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
             else:
                if cp > 1:
                    seq_local = args.seq // cp
                    x_local = x[:, cp_rank*seq_local:(cp_rank+1)*seq_local].contiguous()
                    loss = model.loss(x_local)
                else:
                    loss = model.loss(x)
            with torch.profiler.record_function("backward"):
                loss.backward()
            # CP gradient sync (gpu-sim-fzd): each cp rank trains on a disjoint
            # sequence chunk; combine gradients across cp peers so replicas stay
            # identical and the update matches the full-sequence gradient
            # (gpu-sim-8jg).
            if cp > 1 and cp_group is not None:
                if args.cp_grad_sync == "reducescatter":
                    # Production (gpu-sim-fzd): reduce-scatter the flattened
                    # gradient over cp ranks (each rank gets 1/cp of the SUM),
                    # then all-gather to reconstruct the full gradient. Result is
                    # bit-identical to allreduce+div(cp) up to fp accumulation;
                    # the trace records the reduce_scatter collective so the model's
                    # production-mode byte count (2*(cp-1)/cp * bytes/cp) is
                    # checkable against a real run. Equivalent loss curve.
                    for prm in model.parameters():
                        if prm.grad is not None:
                            flat = prm.grad.flatten().contiguous()
                            # pad to cp divisibility (reuse ZeRO-2 convention)
                            pad = (-flat.numel()) % cp
                            if pad:
                                flat = torch.cat([flat, flat.new_zeros(pad)])
                            shard = torch.empty(flat.numel() // cp, dtype=flat.dtype, device=flat.device)
                            dist.reduce_scatter_tensor(shard, flat, op=dist.ReduceOp.SUM, group=cp_group)
                            # all_gather into a separate buffer (no aliasing with prm.grad)
                            full = [torch.empty_like(shard) for _ in range(cp)]
                            dist.all_gather(full, shard, group=cp_group)
                            prm.grad.copy_(torch.cat(full)[:prm.grad.numel()].view_as(prm.grad))
                            prm.grad.div_(cp)
                else:
                    for prm in model.parameters():
                        if prm.grad is not None:
                            dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM, group=cp_group)
                            prm.grad.div_(cp)
        else:
            # GPipe: all micro-batch forwards, then all backwards. Blocking
            # send/recv in matching order on both sides, so no deadlock: the
            # forward wave runs m=0..M-1 downstream, the backward wave runs
            # m=M-1..0 upstream. Stage outputs are garbage (payload never
            # moves) -- the sends are timing/traffic events, not data.
            inp = x[:, :-1]
            tgt = x[:, 1:]
            outs: list = [None] * mbs
            acts: list = [None] * mbs
            for m in range(mbs):
                sl = slice(m * mb, (m + 1) * mb)
                if pp_idx > 0:
                    a = torch.empty(act_shape, device="cuda")
                    dist.recv(a, src=prev_rank, tag=m)
                    a.requires_grad_(True)
                    acts[m] = a
                    outs[m] = model.run_blocks(a)
                else:
                    outs[m] = model.run_blocks(model.embed_input(inp[sl]))
                if pp_idx < pp - 1:
                    dist.send(outs[m].contiguous(), dst=next_rank, tag=m)
                # Non-leaf stage outputs release their grads unless retained;
                # the last stage needs dLoss/dOuts to send upstream AND the
                # full param backward (autograd.grad w.r.t. outs alone would
                # leave every param grad None and silently skip the DP sync).
                outs[m].retain_grad()
            losses_m: list[float] = []
            for m in range(mbs - 1, -1, -1):
                if pp_idx == pp - 1:
                    sl = slice(m * mb, (m + 1) * mb)
                    l = model.head_loss(outs[m], tgt[sl])
                    losses_m.append(float(l.detach()))
                    l.backward()
                    g = outs[m].grad.contiguous()
                else:
                    g = torch.empty_like(outs[m])
                    dist.recv(g, src=next_rank, tag=m)
                    outs[m].backward(g)
                    if pp_idx > 0:
                        g = acts[m].grad.contiguous()
                if pp_idx > 0:
                    dist.send(g, dst=prev_rank, tag=m)
            loss = None
            losses.append(sum(losses_m) / len(losses_m)
                          if losses_m else float("nan"))
        if dp > 1 and zero == 1:
            # Data-parallel gradient sync: one all_reduce per parameter, which is
            # the traffic DDP would generate (DDP buckets them; un-bucketed is a
            # conservative upper bound and easier to count from the trace).
            # Under TP this reduces ACROSS REPLICAS only, not across the tp group:
            # the tp group's own reductions already happened inside the layers.
            for prm in model.parameters():
                if prm.grad is not None:
                    dist.all_reduce(prm.grad, group=dp_group)
            if dp_mean:
                # The transport SUMS; DDP averages. Divide so the update
                # equals the single-process full-batch mean gradient.
                for prm in model.parameters():
                    if prm.grad is not None:
                        prm.grad.div_(dp)
        elif dp > 1 and zero == 2:
            # ZeRO-2: one reduce_scatter of the FLATTENED grads replaces the
            # per-parameter all_reduces. The shard is a timing event: the local
            # optimizer still steps full params (same timing-only bargain as
            # DDP mode -- the update arithmetic is not simulated).
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if grads:
                flat = torch.cat([gg.reshape(-1) for gg in grads])
                pad = (-flat.numel()) % dp
                if pad:
                    flat = torch.cat([flat, flat.new_zeros(pad)])
                shard = torch.empty(flat.numel() // dp, device="cuda",
                                    dtype=flat.dtype)
                dist.reduce_scatter_tensor(shard, flat, group=dp_group)
                if args.verify_flat:
                    # Reassemble the shards and check against single-process
                    # grads: sum of shard-means divided by dp == full-batch
                    # mean grad, up to fp association order.
                    full = torch.empty(flat.numel(), device="cuda",
                                       dtype=flat.dtype)
                    dist.all_gather_into_tensor(full, shard,
                                                group=dp_group)
                    ref = torch.load(args.verify_flat, weights_only=True)
                    err = (full / dp - ref.to("cuda")).abs().max()
                    print(f"[rank {rank}] zero2-verify step maxerr={err:.3e}",
                          flush=True)
                    assert err < 1e-4 * max(1.0, ref.abs().max()), \
                        f"reassembled grad mismatch: {err:.3e}"
        if args.save_flat:
            # Reference capture: full-batch grads, dp==1 only (a dp>1 run
            # saves averaged shard grads, which is not the reference).
            if dp != 1:
                raise SystemExit("--save-flat needs dp==1")
            sgrads = [p.grad for p in model.parameters()
                      if p.grad is not None]
            torch.save(torch.cat([gg.reshape(-1) for gg in sgrads]).detach(),
                       args.save_flat)
        opt.step()
        if dp > 1 and zero in (1, 2):
            # ZeRO-1/2 keep sharded optimizer states: after the local step the
            # updated params are all_gathered back to full replicas -- the
            # traffic real ZeRO would generate, as a timing event. Gathered
            # into scratch and DISCARDED: the interposer moves no payload, so
            # copying the buffer back would corrupt good local params with
            # uninitialized memory (same reason PP never reads what it recvs).
            # gpu-sim-smt3: this used to SEND THE FULL PARAM and receive dp copies
            # of it (`buf = numel*dp`, `input = prm.contiguous()`), which satisfies
            # all_gather_into_tensor's contract but moves dp x the bytes a real ZeRO
            # param gather moves -- the same dp x error the COST MODEL had before
            # gpu-sim-6r5, mirrored in the harness. A real gather sends each rank's
            # SHARD (numel/dp) and receives the full param (numel):
            for prm in model.parameters():
                if prm.grad is not None:
                    flat = prm.reshape(-1)
                    pad = (-flat.numel()) % dp
                    if pad:
                        flat = torch.cat([flat, flat.new_zeros(pad)])
                    shard = flat.chunk(dp)[rank]
                    buf = torch.empty(flat.numel(), device="cuda",
                                      dtype=prm.dtype)
                    dist.all_gather_into_tensor(buf, shard.contiguous(),
                                                group=dp_group)
        torch.cuda.synchronize()

        delay = 0.0
        if args.delay_ms:
            lo, hi = args.delay_ms
            delay = random.uniform(lo, hi) / 1000.0
            if delay > 0:
                time.sleep(delay)
        if step == n_warm:
            # against steady-state time is how a sweep quietly lies.
            measure_start_ns = time.time_ns()
            # Peak memory is windowed the same way: reset here so the peak
            # covers the measured steps, not CUDA/cuBLAS init.
            torch.cuda.reset_peak_memory_stats()
        if step >= n_warm:
            step_times.append((time.perf_counter() - t0) + delay)
        if not use_pp:
            losses.append(float(loss.detach()))

    if prof is not None:
        prof.__exit__(None, None, None)
        trace_path = f"{args.torch_profile}.rank{rank}.json"
        prof.export_chrome_trace(trace_path)
        print(f"[rank {rank}] torch profile -> {trace_path}", flush=True)

    summary = {
        "rank": rank,
        "world_size": world,
        "host": socket.gethostname(),
        "model": spec.to_dict(),
        "params": spec.param_count(),
        "params_local": (model.local_param_count()
                           if (tp > 1 or use_pp) else spec.param_count()),
        "tp": tp,
        "pp": pp,
        "pp_stage": pp_idx if use_pp else 0,
        "micro_batches": args.micro_batches if use_pp else 1,
        "dp": dp,
        "zero": zero,
        # Measured peak over the measured window (reset at measure_start).
        # This is REAL allocator state -- the honest number. It includes
        # activations, which no ZeRO stage shards; the analytic sharded-state
        # model below is reported alongside so the two can be compared instead
        # of conflated. P = params in bytes (fp32); AdamW holds 2 floats/param.
        "peak_alloc_MB": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "mem_model_MB": _zero_mem_model(spec.param_count() * 4, dp, zero),
        "batch": args.batch,
        "steps": args.steps,
        "warmup_steps": n_warm,
        "clock_anchor": clock_anchor,
        "measure_start_ns": measure_start_ns,
        "step_times_s": step_times,
        "mean_step_s": sum(step_times) / len(step_times),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        # Full per-step series: the tbp gate asserts matching CURVES, not
        # just endpoints (a broken gradient path can match step 1 and stall).
        "losses": losses,
        "synthetic_delay_ms": args.delay_ms,
        "interposer": os.environ.get("LD_PRELOAD", ""),
        "trace": os.environ.get("GPUSIM_TRACE", ""),
    }
    if args.json:
        # One JSON object per rank, appended: a sweep produces one file per
        # (config, rank) without this process having to know about the others.
        with open(args.json, "a") as f:
            f.write(json.dumps(summary) + "\n")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    print(f"[rank {rank}/{world} tp={tp} pp={pp} dp={dp} zero={zero}] "
          f"mean step {summary['mean_step_s']*1000:.1f} ms "
          f"peak {summary['peak_alloc_MB']:.1f} MB "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
