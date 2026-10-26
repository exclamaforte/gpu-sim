"""Does the recompute calibration hold for a GQA block, which is what the models actually are?

The 1.25x time / 0.33-retained figures were measured on nn.MultiheadAttention (MHA, 8 heads). Llama-3 is
GQA(8-of-32 or 8-of-64), where K/V are a fraction of the width, so the activation footprint and what
checkpointing can free are both different. The model charges alpha=0.67 and 1.25x from the MHA numbers, so
this checks whether those transfer.

GQA block: q,o at d; k,v at n_kv_heads*head_dim; SwiGLU MLP at d_ff; attention via
scaled_dot_product_attention(enable_gqa=True).
"""
import json

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEV, DT = "cuda", torch.bfloat16


class GQABlock(torch.nn.Module):
    def __init__(self, d=2048, heads=16, kv_heads=2, hd=128, ff=8192):
        super().__init__()
        self.h, self.kvh, self.hd = heads, kv_heads, hd
        self.q = torch.nn.Linear(d, heads * hd, bias=False, dtype=DT, device=DEV)
        self.k = torch.nn.Linear(d, kv_heads * hd, bias=False, dtype=DT, device=DEV)
        self.v = torch.nn.Linear(d, kv_heads * hd, bias=False, dtype=DT, device=DEV)
        self.o = torch.nn.Linear(heads * hd, d, bias=False, dtype=DT, device=DEV)
        self.g = torch.nn.Linear(d, ff, bias=False, dtype=DT, device=DEV)
        self.u = torch.nn.Linear(d, ff, bias=False, dtype=DT, device=DEV)
        self.dn = torch.nn.Linear(ff, d, bias=False, dtype=DT, device=DEV)
        self.ln1 = torch.nn.LayerNorm(d, dtype=DT, device=DEV)
        self.ln2 = torch.nn.LayerNorm(d, dtype=DT, device=DEV)

    def forward(self, x):
        b, s, d = x.shape
        h = self.ln1(x)
        q = self.q(h).view(b, s, self.h, self.hd).transpose(1, 2)
        k = self.k(h).view(b, s, self.kvh, self.hd).transpose(1, 2)
        v = self.v(h).view(b, s, self.kvh, self.hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, s, d))
        y = self.ln2(x)
        return x + self.dn(F.silu(self.g(y)) * self.u(y))


class Stack(torch.nn.Module):
    def __init__(self, n_layers=8, **kw):
        super().__init__()
        self.blocks = torch.nn.ModuleList([GQABlock(**kw) for _ in range(n_layers)])
        self.ckpt = False

    def forward(self, x):
        for b in self.blocks:
            x = checkpoint(b, x, use_reentrant=False) if self.ckpt else b(x)
        return x


def run(seq, ckpt, d=2048, heads=16, kv_heads=2, ff=8192, layers=8, iters=5):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    net = Stack(layers, d=d, heads=heads, kv_heads=kv_heads, ff=ff); net.ckpt = ckpt
    x = torch.randn(1, seq, d, dtype=DT, device=DEV, requires_grad=True)
    opt = torch.optim.SGD(net.parameters(), lr=0.0)
    for i in range(iters + 1):
        if i == 1:
            torch.cuda.synchronize(); st = torch.cuda.Event(True); en = torch.cuda.Event(True); st.record()
        net(x).sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); en.record(); en.synchronize()
    return {"ms": round(st.elapsed_time(en) / iters, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3)}


if __name__ == "__main__":
    out = {"device": torch.cuda.get_device_name(0), "rows": []}
    for seq in (2048, 4096, 8192):
        base, ck = run(seq, False), run(seq, True)
        row = {"seq": seq, "d": 2048, "heads": 16, "kv_heads": 2,
               "no_ckpt": base, "ckpt": ck,
               "time_ratio": round(ck["ms"] / base["ms"], 3),
               "mem_retained": round(ck["peak_gb"] / base["peak_gb"], 3)}
        out["rows"].append(row); print(row, flush=True)
    print(json.dumps(out, indent=2))
