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
    # Parallelism decomposition. world_size == pp * dp * tp is enforced; the
    # rank layout is rank = pp_idx*(tp*dp) + dp_idx*tp + tp_idx, so each
    # pipeline stage is a contiguous block of tp*dp ranks and TP groups stay
    # contiguous slices inside a stage (the pp==1 layout is unchanged).
    p.add_argument("--tp", type=int, default=1,
                   help="tensor-parallel width; world_size/tp is the data-parallel width")
    p.add_argument("--pp", type=int, default=1,
                   help="pipeline-parallel width: n_layers splits across pp stages")
    p.add_argument("--micro-batches", type=int, default=2,
                   help="GPipe micro-batches per step (must divide --batch)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    return p.parse_args()


def main() -> int:
    args = _parse()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    if world > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    # All ranks share the one REAL GPU: the multi-device fan-out is simulated by
    # the interposer, not by hardware. Asking for device==local_rank is what
    # "invalid device ordinal" means here -- there is no second physical card.
    device_index = int(os.environ.get("HARNESS_DEVICE", "0"))
    torch.cuda.set_device(device_index)

    tp = args.tp
    pp = args.pp
    if world % (pp * tp) != 0:
        raise SystemExit(
            f"world_size {world} is not divisible by pp*tp {pp * tp}")
    dp = world // (pp * tp)
    pp_idx = rank // (tp * dp)
    sub = rank % (tp * dp)
    dp_rank, tp_rank = sub // tp, sub % tp

    spec = ModelSpec(vocab=args.vocab, seq=args.seq, d_model=args.d_model,
                     n_layers=args.layers, n_heads=4, d_ff=args.d_model * 4)
    tp_group = dp_group = None
    if dp > 1:
        # dp groups are strided within a stage: ranks with the same pp stage
        # and tp position form one replica set (pp==1 recovers the old layout).
        for p in range(pp):
            for t in range(tp):
                ranks = [p * tp * dp + d * tp + t for d in range(dp)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    dp_group = g
    if tp > 1:
        # One group per (stage, replica) block of tp ranks. RANK inside a
        # stage is dp_rank * tp + tp_rank, so a group is a contiguous slice
        # and its members share one model replica.
        for p in range(pp):
            for d in range(dp):
                base = p * tp * dp + d * tp
                ranks = list(range(base, base + tp))
                g = dist.new_group(ranks)
                if rank in ranks:
                    tp_group = g
        if pp == 1:
            model = TinyLMTP(spec, tp_size=tp, group=tp_group).cuda()
        else:
            model = PPStage(spec, pp_idx, pp,
                            tp_size=tp, group=tp_group).cuda()
    elif pp > 1:
        model = PPStage(spec, pp_idx, pp).cuda()
    else:
        model = TinyLM(spec).cuda()
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
    # collective is issued and traced); the arithmetic is not. Use this harness
    # to compare COMMUNICATION COST across parallelism configs, never to reason
    # about model convergence. The loss printed in PP mode is garbage.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # One batch shape for every step: the comparison across parallelism configs
    # must not be confounded by a changing workload.
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    x = torch.randint(0, spec.vocab, (args.batch, spec.seq + 1),
                      device="cuda", generator=g)

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
        prev_rank = rank - tp * dp if pp_idx > 0 else None
        next_rank = rank + tp * dp if pp_idx < pp - 1 else None
        act_shape = (mb, spec.seq, spec.d_model)
    for step in range(args.steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if not use_pp:
            loss = model.loss(x)
            loss.backward()
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
            losses_m: list[float] = []
            for m in range(mbs - 1, -1, -1):
                if pp_idx == pp - 1:
                    sl = slice(m * mb, (m + 1) * mb)
                    l = model.head_loss(outs[m], tgt[sl])
                    losses_m.append(float(l.detach()))
                    g = torch.autograd.grad(l, outs[m])[0].contiguous()
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
        if dp > 1:
            # Data-parallel gradient sync: one all_reduce per parameter, which is
            # the traffic DDP would generate (DDP buckets them; un-bucketed is a
            # conservative upper bound and easier to count from the trace).
            # Under TP this reduces ACROSS REPLICAS only, not across the tp group:
            # the tp group's own reductions already happened inside the layers.
            for prm in model.parameters():
                if prm.grad is not None:
                    dist.all_reduce(prm.grad, group=dp_group)
        opt.step()
        torch.cuda.synchronize()

        delay = 0.0
        if args.delay_ms:
            lo, hi = args.delay_ms
            delay = random.uniform(lo, hi) / 1000.0
            if delay > 0:
                time.sleep(delay)
        if step == n_warm:
            # The trace is filtered to this instant, so the collective count and
            # the step times describe the SAME window. Counting warmup traffic
            # against steady-state time is how a sweep quietly lies.
            measure_start_ns = time.time_ns()
        if step >= n_warm:
            step_times.append((time.perf_counter() - t0) + delay)
        if not use_pp:
            losses.append(float(loss.detach()))

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
        "batch": args.batch,
        "steps": args.steps,
        "warmup_steps": n_warm,
        "measure_start_ns": measure_start_ns,
        "step_times_s": step_times,
        "mean_step_s": sum(step_times) / len(step_times),
        "loss_first": losses[0],
        "loss_last": losses[-1],
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
    print(f"[rank {rank}/{world} tp={tp} pp={pp} dp={dp}] "
          f"mean step {summary['mean_step_s']*1000:.1f} ms "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
