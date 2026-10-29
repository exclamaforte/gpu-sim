"""Measure peak activation memory across sequence lengths on the 5090 (gpu-sim-7uz2)."""
import argparse, json, time, subprocess, torch, torch.nn as nn

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

def neighbours():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception as e:
        return f"ERR {e}"

def run_one(seq, batch, n_layers, d, d_ff, n_heads, dtype, steps, warmup):
    torch.cuda.empty_cache()
    model = make_model(n_layers, d, d_ff, n_heads).to(dtype).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(batch, seq, d, device='cuda', dtype=dtype)
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
        "peak_max_bytes": peak_max,
        "peak_median_bytes": peak_med,
        "mem_params_grads_bytes": mem_pg,
        "activation_delta_peak_bytes": activation_mem_peak,
        "bytes_per_token_per_layer_peak": round(bptl_peak, 2),
        "implied_act_factor_peak": round(implied_af_peak, 4),
        "peaks": peaks,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--d-ff", type=int, default=4096)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float32"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seqs", default="2048,4096,8192,16384,32768")
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    seqs = [int(s) for s in args.seqs.split(",")]
    results = []
    for seq in seqs:
        nb = neighbours()
        reps = []
        for rep in range(args.reps):
            try:
                r = run_one(seq, args.batch, args.n_layers, args.d, args.d_ff, args.n_heads, dtype, args.steps, args.warmup)
                r["rep"] = rep
                reps.append(r)
                print(f"seq={seq} rep={rep} af={r['implied_act_factor_peak']} act_delta={r['activation_delta_peak_bytes']}", flush=True)
            except Exception as e:
                print(f"ERROR seq={seq} rep={rep}: {e}", flush=True)
                torch.cuda.empty_cache()
                reps.append({"rep": rep, "error": str(e)})
        na = neighbours()
        agg_af = [r["implied_act_factor_peak"] for r in reps if "implied_act_factor_peak" in r]
        results.append({
            "seq": seq,
            "batch": args.batch,
            "n_layers": args.n_layers,
            "d": args.d,
            "d_ff": args.d_ff,
            "dtype": args.dtype,
            "torch_version": torch.__version__,
            "reps": reps,
            "act_factor_median": (sorted(agg_af)[len(agg_af)//2] if agg_af else None),
            "act_factor_min": (min(agg_af) if agg_af else None),
            "act_factor_max": (max(agg_af) if agg_af else None),
            "spread_fixed_len": ((max(agg_af)-min(agg_af)) if len(agg_af)>1 else 0.0),
            "neighbours_before": nb,
            "neighbours_after": na,
        })
    out = {
        "config": vars(args),
        "model_assumed_act_factor": 2.3,
        "flip_point_N16_seq32768": 2.2769,
        "results": results,
    }
    print("RESULT_JSON_START", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print("RESULT_JSON_END", flush=True)

if __name__ == "__main__":
    main()
