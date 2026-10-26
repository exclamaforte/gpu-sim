"""Measure bytes-per-parameter of a real AdamW step, to check the model's 14 B/param floor.

The model charges fp32 master weights (4 B) + fp32 Adam m and v (8 B) + bf16 grads (2 B) = 14 B/param, and
that term is what makes 405B infeasible at 80 GPUs (70.9 GB floor). If PyTorch's real allocator footprint
differs, the feasibility boundary moves, so this measures it instead of trusting the arithmetic.

Method: one parameter tensor of 5e8 elements (small enough for a 31.4 GiB card at these state costs), bf16
storage, AdamW with default fp32 states, one forward/backward/step. Report peak allocated bytes per param.
"""
import json
import torch

dev = "cuda"
n = 500_000_000
dt = torch.bfloat16


def measure() -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    p = torch.nn.Parameter(torch.randn(n, device=dev, dtype=dt))
    opt = torch.optim.AdamW([p], lr=1e-4)
    base = torch.cuda.memory_allocated()
    x = p.detach().float().sum() * 0.0            # touch the tensor so lazy init cannot hide memory
    del x
    # Forward in bf16 on purpose: an fp32 forward materialises a full fp32 COPY of the parameter
    # (+4 B/param) that a real bf16 model does not pay, and the first version of this script measured
    # 22 B/param because of exactly that.
    loss = (p * p).float().mean()
    loss.backward()
    after_bwd = torch.cuda.memory_allocated()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "params": n,
        "bytes_after_alloc": base,
        "bytes_after_backward": after_bwd,
        "peak_bytes": peak,
        "bytes_per_param_peak": round(peak / n, 3),
        "model_assumption_bytes_per_param": 14.0,
    }


if __name__ == "__main__":
    r = measure()
    r["device"] = torch.cuda.get_device_name(0)
    print(json.dumps(r, indent=2))
