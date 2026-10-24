import os, torch, torch.distributed as dist
rank=int(os.environ["RANK"]); world=int(os.environ["WORLD_SIZE"])
assert world == 2
dist.init_process_group("nccl", rank=rank, world_size=world); torch.cuda.set_device(0)
from tools.torch_parallel.model import ModelSpec, TinyLM, PPStage
spec = ModelSpec(); seed = 13
g = torch.Generator(device="cuda").manual_seed(seed)
x = torch.randint(0, spec.vocab, (8, spec.seq + 1), device="cuda", generator=g)
ref = TinyLM(spec).cuda(); ref.seed_weights(seed)
ref_out = ref(x[:, :-1])
# pp=2: stage rank holds its global layers; activations cross by real send/recv.
stage = PPStage(spec, stage_idx=rank, pp=2, weight_seed=seed).cuda()
xb = x[:, :-1]
if rank == 0:
    a = stage.run_blocks(stage.embed_input(xb))
    dist.send(a.contiguous(), dst=1)
    print("rank0 PP fwd sent", flush=True)
else:
    a = torch.empty(xb.shape[0], xb.shape[1], spec.d_model, device="cuda")
    dist.recv(a, src=0)
    logits = stage.head(stage.lnf(stage.run_blocks(a)))
    # head_loss recomputes cross_entropy; compare logits to reference instead.
    err = (logits - ref_out).abs().max()
    assert torch.equal(logits, ref_out), f"rank1 PP FWD MISMATCH {err:.3e}"
    print(f"rank1 PP fwd PASS exact (err={err:.3e})", flush=True)
dist.barrier(); dist.destroy_process_group()
