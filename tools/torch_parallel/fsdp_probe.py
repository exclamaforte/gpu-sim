"""FSDP viability probe: does FullyShardedDataParallel run under the
timing-only NCCL interposer, and what traffic does it emit?

Expected unknowns (this is what the probe settles):
  - completion: FSDP all_gathers params it then COMPUTES with; gathered
    payload is garbage (uninitialized device memory). Run may complete with
    garbage loss, or abort -- either outcome is recorded, not assumed.
  - traffic: allgather + reducescatter lines with ring_* regimes.
  - memory: torch.cuda.max_memory_allocated per rank (the honest peak).
"""
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

sys.path.insert(0, "/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-8y7.3")
from tools.torch_parallel.model import ModelSpec, TinyLM


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2, "probe is world-2 only"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    spec = ModelSpec()
    model = FSDP(TinyLM(spec).cuda(), device_id=torch.device("cuda:0"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randint(0, spec.vocab, (8, spec.seq + 1), device="cuda", generator=g)
    torch.cuda.reset_peak_memory_stats()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        logits = model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        print(f"[rank {rank}] step {step} loss={float(loss.detach()):.3f}", flush=True)
        loss.backward()
        opt.step()
    print(f"[rank {rank}] peak_alloc_MB={torch.cuda.max_memory_allocated() / 1e6:.1f}",
          flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
