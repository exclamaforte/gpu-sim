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

from tools.torch_parallel.model import ModelSpec, TinyLM


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--delay-ms", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"))
    p.add_argument("--json", type=str, default=None)
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

    spec = ModelSpec(vocab=args.vocab, seq=args.seq, d_model=args.d_model,
                     n_layers=args.layers, n_heads=4, d_ff=args.d_model * 4)
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
    losses: list[float] = []
    for step in range(args.steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x)
        loss.backward()
        if world > 1:
            # Data-parallel gradient sync: one all_reduce per parameter, which is
            # the traffic DDP would generate (DDP buckets them; un-bucketed is a
            # conservative upper bound and easier to count from the trace).
            for prm in model.parameters():
                if prm.grad is not None:
                    dist.all_reduce(prm.grad)
        opt.step()
        torch.cuda.synchronize()

        delay = 0.0
        if args.delay_ms:
            lo, hi = args.delay_ms
            delay = random.uniform(lo, hi) / 1000.0
            if delay > 0:
                time.sleep(delay)
        step_times.append((time.perf_counter() - t0) + delay)
        losses.append(float(loss.detach()))

    summary = {
        "rank": rank,
        "world_size": world,
        "host": socket.gethostname(),
        "model": spec.to_dict(),
        "params": spec.param_count(),
        "batch": args.batch,
        "steps": args.steps,
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
    print(f"[rank {rank}/{world}] mean step {summary['mean_step_s']*1000:.1f} ms "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
