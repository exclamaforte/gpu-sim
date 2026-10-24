import os, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
assert world == 2
dist.init_process_group("nccl", rank=rank, world_size=world); torch.cuda.set_device(0)
n = 1024
# all_gather: rank r contributes arange+ r; gathered order is rank-major.
t = torch.arange(n, dtype=torch.float32, device="cuda") + float(rank)
g = torch.empty(2 * n, dtype=torch.float32, device="cuda")
dist.all_gather_into_tensor(g, t)
want = torch.cat([torch.arange(n, dtype=torch.float32) + float(r) for r in range(2)])
assert torch.equal(g.cpu(), want), f"rank{rank} AG f32 MISMATCH"
print(f"rank{rank} AG f32 exact PASS", flush=True)
b = (torch.arange(n, dtype=torch.float32, device="cuda") + float(rank)).to(torch.bfloat16)
gb = torch.empty(2 * n, dtype=torch.bfloat16, device="cuda")
dist.all_gather_into_tensor(gb, b)
assert torch.allclose(gb.cpu().to(torch.float32), want, rtol=1e-2, atol=1e-2), f"rank{rank} AG bf16 MISMATCH"
print(f"rank{rank} AG bf16 PASS", flush=True)
# reduce_scatter: each rank sends 2n (arange+rank); rank r gets sum of chunk r.
s = torch.arange(2 * n, dtype=torch.float32, device="cuda") + float(rank)
o = torch.empty(n, dtype=torch.float32, device="cuda")
dist.reduce_scatter_tensor(o, s)
want_rs = 2 * torch.arange(n, dtype=torch.float32) + 2 * float(rank) * n + 1.0
assert torch.equal(o.cpu(), want_rs), f"rank{rank} RS MISMATCH err={(o.cpu()-want_rs).abs().max()}"
print(f"rank{rank} RS f32 exact PASS", flush=True)
# send/recv both directions, exact bytes, host-side reference.
m = 512
if rank == 0:
    q = torch.arange(m, dtype=torch.float32, device="cuda")
    dist.send(q, dst=1)
    r = torch.empty(m, dtype=torch.float32, device="cuda")
    dist.recv(r, src=1)
    assert torch.equal(r.cpu(), 2 * torch.arange(m, dtype=torch.float32) + 1.0), "rank0 P2P-back MISMATCH"
    print("rank0 P2P both directions PASS", flush=True)
else:
    r = torch.empty(m, dtype=torch.float32, device="cuda")
    dist.recv(r, src=0)
    assert torch.equal(r.cpu(), torch.arange(m, dtype=torch.float32)), "rank1 P2P-fwd MISMATCH"
    dist.send(2 * torch.arange(m, dtype=torch.float32, device="cuda") + 1.0, dst=0)
    print("rank1 P2P both directions PASS", flush=True)
dist.barrier(); dist.destroy_process_group()
