#!/usr/bin/env python3
"""Measure ACHIEVED bf16 matmul throughput on a real GPU -- the MFU anchor (gpu-sim-wv2).

Why this exists: the cost model's compute term is 6*P*T FLOPs divided by peak FLOPS, with no
efficiency factor. For the published answer (8B, seq 8192, N=8) that works out to an implied
MFU of 76.5% of dense peak. Real training runs report 35-50%. Absolute durations are therefore
optimistic by roughly a factor of two IF the implied efficiency is unrealistic -- and that is a
claim to measure, not to assume.

This script measures the fraction of dense peak achieved by the kernel shapes the model prices.
The FRACTION is the transferable quantity: a 5090 and an H100 differ in peak FLOPS, but neither
reaches 76.5% of its dense peak on large bf16 GEMMs, and the shortfall is the same order.

Usage:
  python3 scripts/measure_mfu.py --peak-tflops 419 --d-model 4096 --seq 8192
"""
import argparse
import json
import sys


def bench(fn, iters, warmup):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters        # seconds per iteration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-tflops", type=float, default=419.0,
                    help="dense bf16 tensor peak for this device (RTX 5090 spec sheet: ~419)")
    ap.add_argument("--seq", type=int, default=8192)
    ap.add_argument("--d-model", type=int, default=4096)
    ap.add_argument("--d-ff-mult", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    peak = args.peak_tflops * 1e12

    # shapes the cost model prices for one training step: projections (M = tokens) and the
    # MLP expansion/contraction
    tokens = args.seq
    d, ffn = args.d_model, args.d_model * args.d_ff_mult
    cases = {
        "qkvo_proj": (tokens, d, d),
        "mlp_up": (tokens, d, ffn),
        "mlp_down": (tokens, ffn, d),
    }
    results = {}
    for label, (m, k, n) in cases.items():
        a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
        t = bench(lambda: a @ b, args.iters, args.warmup)
        flops = 2.0 * m * k * n
        tflops = flops / t / 1e12
        results[label] = {"shape": [m, k, n], "seconds": t, "tflops": tflops,
                          "mfu": tflops * 1e12 / peak}
        print(f"{label:10s} {m}x{k}x{n}  {t*1e3:8.3f} ms  {tflops:7.1f} TFLOP/s  "
              f"MFU {results[label]['mfu']*100:5.1f}%")

    best = max(r["mfu"] for r in results.values())
    print(f"\ndevice {name}; peak assumed {args.peak_tflops} TFLOP/s dense bf16")
    print(f"best achieved MFU across the priced shapes: {best*100:.1f}%")
    payload = {"device": name, "peak_tflops_assumed": args.peak_tflops,
               "results": results, "best_mfu": best}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
