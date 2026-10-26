"""Bytes per parameter for the THREE real optimizer styles, measured (gpu-sim-scup).

The model charges 14 B/param (fp32 master 4 + fp32 Adam m/v 8 + bf16 grads 2). gpu-sim-hz3u's artifact
charges 16 (that plus bf16 weights). My earlier measurement got 10.0 for plain bf16 AdamW. All three can be
right -- they are different training styles -- so this measures each one instead of arguing:

  bf16_adamw   : params bf16, grads bf16, Adam still keeps fp32 m/v. No fp32 master copy.
  amp_classic  : params fp32, autocast to bf16 for compute, GradScaler; grads fp32, Adam fp32.
  bf16_master  : params bf16 + an explicit fp32 master copy updated by the optimizer (DeepSpeed/FSDP style).

Peak allocated bytes / param for one fwd+bwd+step over a 5e8-parameter tensor.
"""
import json

import torch

N = 500_000_000
DEV = "cuda"


def _loss(p, style):
    """Reduce WITHOUT any parameter-sized temporary.

    Every earlier version of this used (p*p) or (p*p).sum(), which retains a parameter-sized fp32
    intermediate until backward and inflates EVERY row by the same 4 B/param. p.sum() allocates only a scalar
    and the gradient of a sum is a broadcast, so the measurement sees exactly the state a training step needs.
    """
    return p.sum()


def measure(style: str) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dt = torch.float32 if style == "amp_classic" else torch.bfloat16
    p = torch.nn.Parameter(torch.randn(N, device=DEV, dtype=dt))
    opt = None if style == "bf16_master" else torch.optim.AdamW([p], lr=1e-4)
    master = None
    if style == "bf16_master":
        master = p.detach().float().clone()           # explicit fp32 master
    loss = _loss(p, style)
    loss.backward()
    if style == "bf16_master":
        # DeepSpeed-style: ONLY the fp32 master carries optimizer state; the bf16 param never does.
        # The first version of this measured 24 B/param because it built a second AdamW over the bf16 param
        # as well, i.e. it counted two sets of moments.
        with torch.no_grad():
            master.grad = p.grad.float()
            opt_m = torch.optim.AdamW([master], lr=1e-4)
            opt_m.step()
            p.copy_(master)
            del opt_m
    else:
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    steady = torch.cuda.memory_allocated()      # after step, temporaries freed: what training RESIDES at
    return {"style": style, "params": N, "peak_gb": round(peak / 1e9, 3),
            "peak_bytes_per_param": round(peak / N, 2),
            "steady_gb": round(steady / 1e9, 3),
            "steady_bytes_per_param": round(steady / N, 2)}


if __name__ == "__main__":
    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "results": [measure(s) for s in ("bf16_adamw", "amp_classic", "bf16_master")]}
    print(json.dumps(out, indent=2))
