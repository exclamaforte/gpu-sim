import os, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
dist.init_process_group("nccl", rank=rank, world_size=world); torch.cuda.set_device(0)
n = 1024
# f32: exact cross-rank sum, asserted against a HOST-side reference built on CPU.
t = torch.arange(n, dtype=torch.float32, device="cuda") + float(rank)
dist.all_reduce(t)
want = 2 * torch.arange(n, dtype=torch.float32) + float(world - 1)
assert torch.equal(t.cpu(), want), f"rank{rank} f32 MISMATCH err={(t.cpu()-want).abs().max()}"
print(f"rank{rank} f32 exact-sum PASS", flush=True)
# bf16: same, within conversion tolerance.
b = (torch.arange(n, dtype=torch.float32, device="cuda") + float(rank)).to(torch.bfloat16)
dist.all_reduce(b)
got = b.cpu().to(torch.float32)
assert torch.allclose(got, want, rtol=1e-2, atol=1e-2), f"rank{rank} bf16 MISMATCH err={(got-want).abs().max()}"
print(f"rank{rank} bf16 sum PASS", flush=True)
dist.barrier(); dist.destroy_process_group()
