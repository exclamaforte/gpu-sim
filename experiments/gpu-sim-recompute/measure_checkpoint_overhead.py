"""Measure what activation checkpointing actually costs, to calibrate the `recompute` term.

The model is about to gain a recompute knob (gpu-sim-hz3u) and was told to cite a published overhead.
A measured number from the one GPU in this lab is better than a cited rule of thumb, provided the scope is
stated: one RTX 5090, bf16, no tensor/pipeline parallelism, so this measures the COMPUTE overhead of
recomputation and the ACTIVATION memory it saves -- not the distributed interference.

Method: a Llama-style decoder block at two widths/seqs. Time forward+backward with
torch.utils.checkpoint(use_reentrant=False) and without, report time ratio and peak-memory ratio.
"""
import json
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)
dev = "cuda"
dt = torch.bfloat16


class Stack(nn.Module):
    """A real stack: checkpointing saves the INTRA-layer activations of every layer, which a single block
    cannot show (one block's peak is dominated by its own weights)."""

    def __init__(self, d, heads, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(n_layers)])

    apply_ckpt = False        # set by run(); the standard pattern is one checkpoint per BLOCK

    def forward(self, x):
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if Stack.apply_ckpt else b(x)
        return x


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dtype=dt, device=dev)
        self.ln1 = nn.LayerNorm(d, dtype=dt, device=dev)
        self.ln2 = nn.LayerNorm(d, dtype=dt, device=dev)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d, dtype=dt, device=dev), nn.GELU(),
                                nn.Linear(4 * d, d, dtype=dt, device=dev))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


def run(d, heads, seq, ckpt, iters=6, n_layers=8):
    block = Stack(d, heads, n_layers)
    Stack.apply_ckpt = ckpt
    x = torch.randn(1, seq, d, dtype=dt, device=dev, requires_grad=True)
    opt = torch.optim.SGD(block.parameters(), lr=0.0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for i in range(iters + 2):          # 2 warmup
        if i == 2:
            torch.cuda.synchronize()
            start = torch.cuda.Event(True); end = torch.cuda.Event(True)
            start.record()
        out = block(x)
        out.sum().backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    end.record(); end.synchronize()
    ms = start.elapsed_time(end) / iters
    return {"ms_per_iter": round(ms, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}


res = {}
for (d, heads, seq) in ((1024, 8, 8192), (2048, 16, 4096), (1024, 8, 4096)):
    key = f"d{d}_h{heads}_seq{seq}_L8"
    base = run(d, heads, seq, False)
    ck = run(d, heads, seq, True)
    res[key] = {"no_recompute": base, "recompute": ck,
                "time_ratio": round(ck["ms_per_iter"] / base["ms_per_iter"], 3),
                "mem_ratio": round(ck["peak_gb"] / base["peak_gb"], 3)}
    print(key, res[key], flush=True)

print(json.dumps({"device": torch.cuda.get_device_name(0), "dtype": "bf16", "results": res}, indent=2))
