"""Timeout validation: rank 1 dies mid-collective; rank 0 must print the
(op, k, rank, nranks, arrived) dump and abort -- never hang silently."""
import os, time, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", rank=rank, world_size=world); torch.cuda.set_device(0)
if rank == 1:
    dist.destroy_process_group()
    print("[kill-test] rank 1 exiting before the collective", flush=True)
    os._exit(0)
time.sleep(3)  # let rank 1 die first: the collective truly loses its peer
t = torch.ones(16, dtype=torch.float32, device="cuda")
print("[kill-test] rank 0 entering all_reduce with a dead peer; "
      "expect barrier-timeout dump, then SIGABRT", flush=True)
dist.all_reduce(t)
print("[kill-test] UNREACHABLE: collective returned without peer", flush=True)
