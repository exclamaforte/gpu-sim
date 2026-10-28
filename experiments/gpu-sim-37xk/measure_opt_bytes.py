#!/usr/bin/env python3
"""Measure real optimizer bytes/param on a 5090 for AdamW modes."""
import argparse, json, torch, torch.nn as nn

def make_model(n_params_target, dtype):
    # stack Linear layers to ~n_params_target params
    hidden = 4096
    # params per layer: hidden*hidden + hidden (bias) ~ 16.78M
    per_layer = hidden * hidden + hidden
    n_layers = max(1, n_params_target // per_layer)
    layers = []
    for i in range(n_layers):
        layers.append(nn.Linear(hidden, hidden, bias=True))
    # first layer input adjust not needed; keep square
    model = nn.Sequential(*layers)
    model = model.to(dtype=dtype)
    return model, n_layers * per_layer

def run_mode(mode, dtype, n_params_target, steps=8, warmup=2):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, n_params = make_model(n_params_target, dtype)
    model = model.cuda()
    # params+grads baseline: forward/backward once without optimizer
    x = torch.randn(4, 4096, device='cuda', dtype=dtype)
    out = model(x)
    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    mem_params_grads = torch.cuda.memory_allocated()
    # create optimizer
    if mode == 'foreach_false':
        foreach, fused = False, False
    elif mode == 'foreach_true':
        foreach, fused = True, False
    else:
        foreach, fused = None, True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=foreach, fused=fused)
    # optimizer state init: step once
    opt.step()
    torch.cuda.synchronize()
    mem_after_init = torch.cuda.memory_allocated()
    opt.zero_grad(set_to_none=True)
    # warmup
    for _ in range(warmup):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    peaks = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated())
    torch.cuda.synchronize()
    peak_max = max(peaks)
    peak_med = sorted(peaks)[len(peaks)//2]
    # dtype info
    p0 = next(model.parameters())
    grad_dtype = None
    for p in model.parameters():
        if p.grad is not None:
            grad_dtype = str(p.grad.dtype); break
    # master weights materialised? check optimizer state has 'params' with fp32 for bf16 model
    master_materialised = False
    # AdamW stores state per param; inspect first param state
    # master weights: torch AdamW does not materialise fp32 master weights for bf16 params
    master_materialised = (str(p0.dtype) == 'torch.float32')
    result = {
        "mode": mode,
        "param_dtype": str(p0.dtype),
        "grad_dtype": grad_dtype,
        "n_params": n_params,
        "n_layers": len(list(model.children())) if hasattr(model,'children') else None,
        "mem_params_grads_bytes": mem_params_grads,
        "mem_after_opt_init_bytes": mem_after_init,
        "peak_step_max_bytes": peak_max,
        "peak_step_median_bytes": peak_med,
        "bytes_per_param_params_grads": round(mem_params_grads / n_params, 6),
        "bytes_per_param_after_init": round(mem_after_init / n_params, 6),
        "bytes_per_param_peak_max": round(peak_max / n_params, 6),
        "bytes_per_param_peak_median": round(peak_med / n_params, 6),
        "peak_steps": peaks,
        "master_weights_materialised": master_materialised,
        "torch_version": torch.__version__,
    }
    del model, opt
    torch.cuda.empty_cache()
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-params", type=int, default=200_000_000)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float32"])
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    out = {"n_params_target": args.n_params, "dtype": args.dtype, "steps": args.steps, "modes": []}
    for mode in ["foreach_false", "foreach_true", "fused"]:
        # AdamW foreach flag: False/True/None(fused uses fused=True, foreach=None)
        # handled inside run_mode by mode string
        r = run_mode(mode, dtype, args.n_params, steps=args.steps)
        out["modes"].append(r)
        print(json.dumps(r, indent=2))
    print("RESULT_JSON_START")
    print(json.dumps(out, indent=2))
    print("RESULT_JSON_END")

if __name__ == "__main__":
    main()