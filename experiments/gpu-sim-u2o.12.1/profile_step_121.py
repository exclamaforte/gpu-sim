#!/usr/bin/env python3
"""gpu-sim-u2o.12.1: step-time/MFU profile of the CURRENT v1 shape.

9.1's dual-head encoder as merged (d512/8h/12L/dff2048, numeric_dim=33,
dual-head, concat fusion) at batch 32 x seq 512 with the SHARED NLL loss.
Synchronized timing (Sep-28 defect #1: no bare time.time() around launches).
Input shapes match TTDataset batches (Sep-28 defect #3); values are random
(step time is value-independent). FLOPs = 6*N*S*B upper bound, N read from
the model, not retyped. Data-gen side read from 6.7.1's committed artifact.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))

import torch  # noqa: E402

from epic_u2o.model import create_model, gaussian_nll_with_floor  # noqa: E402
from epic_u2o.hw_5090 import PEAK_FP32_TFLOPS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    torch.manual_seed(0)
    model = create_model(vocab_size=256, max_seq_len=512, d_model=512,
                         n_heads=8, n_layers=12, d_ff=2048, numeric_dim=33,
                         use_numeric=True, dual_head=True,
                         fusion_type="concat")
    n_params = sum(p.numel() for p in model.parameters())
    assert len(model.encoder.layers) == 12, "v1 shape is 12 layers"
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    B, S = args.batch, 512

    def batch():
        return (torch.randint(1, 256, (B, S), device=device),
                torch.ones(B, S, dtype=torch.long, device=device),
                torch.randn(B, 33, device=device),
                torch.randn(B, 1, device=device),
                torch.rand(B, 1, device=device) * 0.3)

    def step():
        ids, mask, num, y, sig = batch()
        mu, vp = model(ids, mask, num)
        loss = gaussian_nll_with_floor(mu, vp, y, sig)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    for _ in range(args.warmup):
        step()
    times = []
    for i in range(args.steps):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        print(f"GPQUEUE_PROGRESS done={i + 1} total={args.steps} "
              f"unit=steps", flush=True)

    med = statistics.median(times)
    toks = B * S
    flops = 6 * n_params * S * B  # upper bound (numeric MLP extra); MFU a lower bound
    tflops = flops / med / 1e12
    mfu = 100.0 * tflops / PEAK_FP32_TFLOPS
    gen = json.loads((REPO / "experiments" / "gpu-sim-u2o.6.7.1"
                      / "batch_summary.json").read_text())["throughput"]
    gen_exs = gen["examples_per_sec"]
    train_exs = B / med
    summary = {
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "cuda": cuda,
        "model": {"d_model": 512, "n_heads": 8, "n_layers": 12,
                  "d_ff": 2048, "numeric_dim": 33, "dual_head": True,
                  "fusion": "concat", "n_params": n_params},
        "batch": B, "seq": S, "warmup": args.warmup, "steps": args.steps,
        "step_time_s_median": med,
        "step_time_s_all": times,
        "tokens_per_step": toks,
        "flops_per_step_6NSB_upper_bound": flops,
        "achieved_tflops": tflops,
        "mfu_pct_vs_intree_peak": mfu,
        "peak_tflops_used": PEAK_FP32_TFLOPS,
        "peak_note": ("in-tree hw_5090.PEAK_FP32_TFLOPS, flagged UNSOURCED "
                      "in-tree; achieved TFLOP/s is the trustworthy column"),
        "train_examples_per_sec": train_exs,
        "datagen_examples_per_sec": gen_exs,
        "datagen_source": ("experiments/gpu-sim-u2o.6.7.1/batch_summary.json"
                           ".throughput.examples_per_sec (read, not retyped)"),
        "train_faster_than_gen_x": train_exs / gen_exs,
    }
    (out / "step_profile.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("step_time_s_median", "achieved_tflops",
                       "mfu_pct_vs_intree_peak", "train_examples_per_sec",
                       "datagen_examples_per_sec")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
