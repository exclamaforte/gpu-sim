#!/usr/bin/env python3
"""Measure activation memory (bytes/token/layer) on a 5090 for seq 8192 vs 32768."""
import argparse, json, torch, torch.nn as nn

def make_block(d, d_ff, n_heads):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        def forward(self, x):
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + a
            h = self.ln2(x)
            x = x + self.ffn(h)
            return x
    return Block()

def make_model(n_layers, d, d_ff, n_heads):
    blocks = nn.ModuleList([make_block(d, d_ff, n_heads) for _ in range(n_layers)])
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = blocks
            self.ln = nn.LayerNorm(d)
        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return self.ln(x)
    return M()

def run(seq, batch, n_layers, d, d_ff, n_heads, dtype, steps=5, warmup=2):
    torch.cuda.empty_cache()
    model = make_model(n_layers, d, d_ff, n_heads).to(dtype).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(batch, seq, d, device='cuda', dtype=dtype)
    # warmup
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        out = model(x)
        out.sum().backward()
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    peaks = []
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        out = model(x)
        out.sum().backward()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    mem_pg = torch.cuda.memory_allocated()
    peak_max = max(peaks)
    peak_med = sorted(peaks)[len(peaks)//2]
    dtype_bytes = 2 if dtype == torch.bfloat16 else 4
    per_token_width = 3 * d + d_ff
    activation_mem_peak = peak_max - mem_pg
    bptl_peak = activation_mem_peak / (batch * seq * n_layers)
    implied_af_peak = bptl_peak / (dtype_bytes * per_token_width)
    del model, x
    torch.cuda.empty_cache()
    return {
        "seq": seq,
        "batch": batch,
        "n_layers": n_layers,
        "d": d,
        "d_ff": d_ff,
        "n_params": n_params,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "peak_max_bytes": peak_max,
        "peak_median_bytes": peak_med,
        "mem_params_grads_bytes": mem_pg,
        "activation_delta_peak_bytes": activation_mem_peak,
        "bytes_per_token_per_layer_peak": round(bptl_peak, 2),
        "per_token_width_base": per_token_width,
        "dtype_bytes": dtype_bytes,
        "implied_act_factor_peak": round(implied_af_peak, 3),
        "peaks": peaks,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--d-ff", type=int, default=4096)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float32"])
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    results = []
    for seq in [8192, 32768]:
        for batch in [1, 2]:
            try:
                r = run(seq, batch, args.n_layers, args.d, args.d_ff, args.n_heads, dtype, args.steps, args.warmup)
                results.append(r)
                print(json.dumps(r, indent=2), flush=True)
            except Exception as e:
                print(f"ERROR seq={seq} batch={batch}: {e}", flush=True)
                torch.cuda.empty_cache()
    out = {"config": vars(args), "results": results, "model_assumed_act_factor": 2.3}
    print("RESULT_JSON_START", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print("RESULT_JSON_END", flush=True)

if __name__ == "__main__":
    main()
