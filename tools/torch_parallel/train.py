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

from tools.torch_parallel.model import ModelSpec, TinyLM, TinyLMTP


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
    # Parallelism decomposition. world_size == dp * tp is enforced; there is no
    # pipeline axis yet, so pp is fixed at 1.
    p.add_argument("--tp", type=int, default=1,
                   help="tensor-parallel width; world_size/tp is the data-parallel width")
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
    if world % tp != 0:
        raise SystemExit(f"world_size {world} is not divisible by tp {tp}")
    dp = world // tp
    tp_rank, dp_rank = rank % tp, rank // tp

    spec = ModelSpec(vocab=args.vocab, seq=args.seq, d_model=args.d_model,
                     n_layers=args.layers, n_heads=4, d_ff=args.d_model * 4)
    tp_group = dp_group = None
    if dp > 1:
        for base in range(0, world, dp):
            pass
        # dp groups are strided: ranks with the same tp_rank form one replica set
        for t in range(tp):
            ranks = [d * tp + t for d in range(dp)]
            g = dist.new_group(ranks)
            if rank in ranks:
                dp_group = g
    if tp > 1:
        # One group per contiguous block of tp ranks. RANK = dp_rank * tp + tp_rank,
        # so a group is a contiguous slice and its members share one model replica.
        for base in range(0, world, tp):
            ranks = list(range(base, base + tp))
            g = dist.new_group(ranks)
            if rank in ranks:
                tp_group = g
        model = TinyLMTP(spec, tp_size=tp, group=tp_group).cuda()
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
    # is NOT the true sum across ranks. Timing and traffic are real (every
    # collective is issued and traced); the arithmetic is not. Use this harness
    # to compare COMMUNICATION COST across parallelism configs, never to reason
    # about model convergence.
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
    for step in range(args.steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x)
        loss.backward()
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
        losses.append(float(loss.detach()))

    summary = {
        "rank": rank,
        "world_size": world,
        "host": socket.gethostname(),
        "model": spec.to_dict(),
        "params": spec.param_count(),
        "params_local": model.local_param_count() if tp > 1 else spec.param_count(),
        "tp": tp,
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
    print(f"[rank {rank}/{world} tp={tp} dp={dp}] "
          f"mean step {summary['mean_step_s']*1000:.1f} ms "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
