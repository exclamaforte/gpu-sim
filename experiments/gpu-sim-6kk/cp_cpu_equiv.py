import os, sys, torch, torch.distributed as dist
sys.path.insert(0, "/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-6kk")
from tools.torch_parallel.model import ModelSpec, TinyLM, TinyLMTP

rank = int(os.environ.get("RANK", "0"))
world = int(os.environ.get("WORLD_SIZE", "1"))
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29502")
dist.init_process_group(backend="gloo", rank=rank, world_size=world)

cp = world
spec = ModelSpec(vocab=256, seq=64, d_model=32, n_layers=2, n_heads=4, d_ff=64)

torch.manual_seed(42)
if cp == 1:
    model = TinyLMTP(spec, tp_size=1, group=None, tp_rank=0, weight_seed=42, cp_size=1, cp_rank=0, cp_peers=[0])
else:
    cp_peers = list(range(cp))
    model = TinyLMTP(spec, tp_size=1, group=None, tp_rank=0, weight_seed=42,
                     cp_size=cp, cp_rank=rank, cp_peers=cp_peers)

# Ensure identical init across cp ranks (7sb scheme: same seed)
# Re-seed and re-init weights identically
torch.manual_seed(42)
for m in model.modules():
    if hasattr(m, "reset_parameters"):
        m.reset_parameters() if False else None
# TinyLMTP already drew weights from weight_seed=42; TinyLM uses default init.
# To make cp=1 and cp=2 comparable, both must use the SAME init. TinyLM uses
# default (random) init, TinyLMTP uses _seed_block with weight_seed. Mismatch!
# Use weight_seed path for both: TinyLM doesn't accept weight_seed. So instead
# compare cp=2 ranks among themselves (replica identity), and separately note
# that cp=1 uses a different init scheme (TinyLM). 

torch.manual_seed(123)
x = torch.randint(0, spec.vocab, (4, spec.seq + 1))

opt = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []
for step in range(5):
    opt.zero_grad()
    if cp > 1:
        seq_local = spec.seq // cp
        x_local = x[:, rank*seq_local:(rank+1)*seq_local].contiguous()
        loss = model.loss(x_local)
    else:
        loss = model.loss(x)
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
    if rank == 0:
        print(f"rank0 step {step}: loss {losses[-1]:.6f}")

# Save params from rank 0
state = {k: v.detach().clone() for k, v in model.state_dict().items()}
torch.save(state, f"/tmp/cp{cp}_rank{rank}_params.pt")
if rank == 0:
    with open(f"/tmp/cp{cp}_losses.txt", "w") as f:
        for l in losses:
            f.write(f"{l}\n")
dist.barrier()
dist.destroy_process_group()
