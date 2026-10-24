import os, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
assert world == 2
dist.init_process_group("nccl", rank=rank, world_size=world); torch.cuda.set_device(0)
from tools.torch_parallel.model import ModelSpec, TinyLM, TinyLMTP
spec = ModelSpec(); seed = 11
g = torch.Generator(device="cuda").manual_seed(seed)
x = torch.randint(0, spec.vocab, (8, spec.seq + 1), device="cuda", generator=g)
# Reference: unsharded, seeded by construction.
ref = TinyLM(spec).cuda(); ref.seed_weights(seed)
ref_out = ref(x[:, :-1])
# Sharded: full-draw + slice, same seed.
tp_group = dist.new_group([0, 1])
tp = TinyLMTP(spec, tp_size=2, group=tp_group, tp_rank=rank,
              weight_seed=seed).cuda()
out = tp(x[:, :-1])
err = (out - ref_out).abs().max()
assert torch.allclose(out, ref_out, rtol=1e-4, atol=1e-4), f"rank{rank} TP FWD MISMATCH {err:.3e}"
print(f"rank{rank} TP fwd PASS err={err:.3e}", flush=True)
# 20 SGD steps, same data/lr on both: trajectories must match AND fall.
opt_r = torch.optim.SGD(ref.parameters(), lr=0.01)
opt_t = torch.optim.SGD(tp.parameters(), lr=0.01)
prev = None
for step in range(20):
    lr_ = ref.loss(x); lr_.backward(); opt_r.step(); opt_r.zero_grad()
    lt_ = tp.loss(x); lt_.backward(); opt_t.step(); opt_t.zero_grad()
    d = abs(float(lr_) - float(lt_))
    assert d < 1e-4 * max(1.0, abs(float(lr_))), f"rank{rank} step{step} {float(lr_):.4f} vs {float(lt_):.4f}"
    if prev is not None:
        assert float(lt_) <= prev + 1e-6, f"rank{rank} step{step} rose {prev:.4f}->{float(lt_):.4f}"
    prev = float(lt_)
print(f"rank{rank} TP 20-step PASS {float(lt_):.4f} monotone", flush=True)
dist.barrier(); dist.destroy_process_group()
