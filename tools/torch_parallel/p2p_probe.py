"""2-rank p2p probe: verify ncclSend/ncclRecv emit trace lines (master 5f6fe05)."""
import json
import os
import sys

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2
    dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    torch.cuda.set_device(0)
    t = torch.full((8, 8), float(rank + 1), device="cuda")
    if rank == 0:
        dist.send(t, dst=1)
    else:
        dist.recv(t, src=0)
    # receiver keeps init value under the timing-only interposer
    print(f"[rank {rank}] sum after p2p: {float(t.sum())}")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
