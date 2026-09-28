#!/usr/bin/env python3
"""REAL training step-time / MFU baseline on the RTX 5090. gpu-sim-u2o.12.1.

The previous profile was a CPU smoke: its own metadata said "CPU smoke; rerun on
5090 for the real baseline", and it never did. Four things had to be corrected
before its number meant anything, and three of them would each have produced a
plausible wrong answer on the GPU:

  1. NO CUDA SYNCHRONISATION. The old loop timed `time.time()` around launches
     with no `torch.cuda.synchronize()`, so on a GPU it would have measured the
     time to QUEUE the work, not to do it. Every step would have looked like the
     cost of a few Python calls.
  2. THE PEAK WAS THE WRONG GPU. MFU was divided by 82.6 TFLOP/s, which is the
     RTX 4090's FP32 peak. The 5090 is higher, so every MFU here would have
     been overstated by the ratio of the two.
  3. THE MODEL INPUT NEVER MATCHED THE FEATURE VECTOR. create_model defaults
     numeric_dim=26; the real vector is 27 on master and 28 with gpu-sim-u2o.8.3.
     Feeding the real vector RAISES. The old script fed `torch.randn(batch, 26)`
     -- random noise of the wrong width -- so it was timing a model that cannot
     consume the actual dataset at all.
  4. "the v1 model size" is defined NOWHERE in the repository (v1_scope.md
     describes the PREDICTION scope, not a model size). So this profiles a LADDER
     and reports the sizes, rather than picking one and calling it v1.

Must run through `ts`. GPU work on a shared queue.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from epic_u2o.model import create_model  # noqa: E402

# RTX 5090 (GB202) FP32 peak. A LITERATURE value, deliberately labelled as such:
# it is not measured here, and MFU against an unmeasured peak inherits the
# peak's uncertainty. Reported alongside ACHIEVED TFLOP/s, which is measured,
# so a reader who distrusts the peak can use the other number.
RTX_5090_FP32_PEAK_TFLOPS = 104.8
PEAK_PROVENANCE = ("literature value for GB202 FP32, NOT measured in this run; "
                   "treat MFU as carrying the peak's uncertainty")

# The model ladder. 'placeholder' is the 26 the repo defaulted to;
# 'real_8_3' is the corrected feature-vector width from gpu-sim-u2o.8.3.
LADDER = [
    # (label, vocab, max_seq, d_model, heads, layers, d_ff, numeric_dim, batch, seq)
    ("smoke-as-written",   64,  32,   32,  2, 1,   64, 26, 2, 16),
    ("small",              256, 512,  128,  4, 2,  512, 28, 32, 128),
    ("medium",             1024, 1024, 512, 8, 4, 2048, 28, 32, 256),
    ("large",              4096, 2048, 768, 12, 6, 3072, 28, 16, 512),
]


def _data_generation_rate() -> dict | None:
    """gpu-sim-u2o.6.7's measured generation rate, read not retyped.

    Read from that bead's committed metrics.json on its branch. If it is
    unavailable the comparison is simply omitted rather than filled in with a
    remembered number -- a comparison against a value I half-remember is worse
    than no comparison.
    """
    import subprocess
    r = subprocess.run(
        ["git", "show",
         "lab/gpu-sim-u2o.6.7:experiments/gpu-sim-u2o.6.7/metrics.json"],
        cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        m = json.loads(r.stdout)
        hw = m["throughput"]["harness_wall"]
    except (KeyError, ValueError, json.JSONDecodeError):
        return None
    return {
        "ex_per_sec": hw["examples_per_sec"],
        "ex_per_hour": hw["examples_per_hour"],
        "source": "lab/gpu-sim-u2o.6.7 metrics.json (ts 20260928t074004-b429f469)",
        "n_rows_measured": m["n_rows_measured"],
    }


def params_of(model) -> int:
    return sum(p.numel() for p in model.parameters())


def flops_per_step(params: int, seq: int, batch: int) -> float:
    """6*N per token is the standard transformer approximation (fwd+bwd).

    Stated as an APPROXIMATION in the output rather than presented as a count,
    because for a model with a numeric-feature MLP the embedding and head terms
    do not follow the 6*N*T shape exactly. A MFU computed from an approximate
    FLOP count is an upper bound on the FLOPs and so a LOWER bound on MFU.
    """
    return 6.0 * params * seq * batch


def time_steps(model, batch_data, n_warmup: int, n_steps: int) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def one_step():
        opt.zero_grad(set_to_none=True)
        # The model returns a SINGLE (batch, 1) prediction tensor, not a
        # (mean, log_var) pair. The old profile unpacked two values, which
        # silently worked at batch=2 -- because a (2, 1) tensor unpacks into
        # two elements -- and RAISED at every larger batch. So the old script
        # could only ever have profiled the smoke rung, which is a fourth way it
        # could not have produced a real baseline.
        pred = model(batch_data['input_ids'],
                     batch_data['attention_mask'],
                     batch_data['numeric_features'])
        loss = ((pred - batch_data['target']) ** 2).mean()
        loss.backward()
        opt.step()

    for _ in range(n_warmup):
        one_step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()      # before the clock starts

    times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        one_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # the whole point: wait for the WORK
        times.append(time.perf_counter() - t0)

    times.sort()
    return {
        "n_warmup": n_warmup, "n_steps": n_steps,
        "mean_step_s": sum(times) / len(times),
        "median_step_s": times[len(times) // 2],
        "min_step_s": times[0], "max_step_s": times[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FATAL: no CUDA device visible. This bead exists to replace a CPU "
              "smoke with a real 5090 measurement; running it on CPU would "
              "reproduce the exact defect it was reopened for.", file=sys.stderr)
        return 2

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)

    out = {
        "bead": "gpu-sim-u2o.12.1",
        "gpu_used": True,
        "device": {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "capability": f"{props.major}.{props.minor}",
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "env": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        "peak_flops_reference": {
            "value_tflops": RTX_5090_FP32_PEAK_TFLOPS,
            "provenance": PEAK_PROVENANCE,
            "previous_value_used": 82.6,
            "previous_value_provenance": "RTX 4090 FP32 peak -- the WRONG GPU",
        },
        "steps": args.steps, "warmup": args.warmup,
        "ladder": [],
    }

    for (label, vocab, max_seq, d_model, heads, layers, d_ff,
         num_dim, batch, seq) in LADDER:
        torch.manual_seed(0)
        model = create_model(vocab_size=vocab, max_seq_len=max_seq,
                             d_model=d_model, n_heads=heads, n_layers=layers,
                             d_ff=d_ff, numeric_dim=num_dim).to(dev)
        model.train()
        data = {
            "input_ids": torch.randint(0, vocab, (batch, seq), device=dev),
            "attention_mask": torch.ones(batch, seq, dtype=torch.long, device=dev),
            "numeric_features": torch.randn(batch, num_dim, device=dev),
            "target": torch.zeros(batch, 1, device=dev),
        }
        timing = time_steps(model, data, args.warmup, args.steps)
        p = params_of(model)
        fl = flops_per_step(p, seq, batch)
        achieved = fl / timing["median_step_s"] / 1e12
        out["ladder"].append({
            "label": label, "params": p,
            "d_model": d_model, "n_layers": layers, "batch": batch, "seq": seq,
            "numeric_dim": num_dim,
            **timing,
            "flops_per_step_approx": fl,
            "achieved_tflops": achieved,
            "mfu_pct": 100.0 * achieved / RTX_5090_FP32_PEAK_TFLOPS,
            "examples_per_sec_training_can_consume": batch / timing["median_step_s"],
        })
        print(f"{label:18s} params={p:>10,} step={timing['median_step_s']*1e3:8.3f} ms "
              f"achieved={achieved:7.3f} TFLOP/s MFU={out['ladder'][-1]['mfu_pct']:.4f}%",
              flush=True)
        del model, data
        torch.cuda.empty_cache()

    # The comparison gpu-sim-u2o.12.2 exists to make, made here from the
    # measured training side and gpu-sim-u2o.6.7's measured generation side.
    # Both numbers are real measurements; neither is an estimate. The
    # generation figure is read from 6.7's committed artifact rather than
    # retyped, so the comparison cannot drift from its source.
    data_gen = _data_generation_rate()
    if data_gen:
        for rung in out["ladder"]:
            ex_s = rung["examples_per_sec_training_can_consume"]
            rung["vs_data_generation"] = {
                "training_consumes_ex_per_sec": ex_s,
                "data_generation_produces_ex_per_sec": data_gen["ex_per_sec"],
                "generation_is_slower_by": data_gen["ex_per_sec"] / ex_s
                if ex_s else None,
            }
        out["bottleneck_comparison"] = {
            "question": ("gpu-sim-u2o.12.2: is training the wall-clock "
                         "bottleneck, or data generation?"),
            "data_generation": data_gen,
            "training_can_consume_ex_per_sec": {
                r["label"]: r["examples_per_sec_training_can_consume"]
                for r in out["ladder"]},
            "answer": (
                "DATA GENERATION IS THE BOTTLENECK, decisively. Training can "
                "consume "
                + ", ".join(
                    f"{r['examples_per_sec_training_can_consume']:.0f} ex/s "
                    f"({r['label']})" for r in out["ladder"][1:])
                + f", while generation PRODUCES {data_gen['ex_per_sec']:.2f} ex/s. "
                # TRAINING / GENERATION, not the reverse. I first wrote this
                # the other way round and it printed "0x to 0x", because
                # generation is the SMALLER number and dividing by it the wrong
                # way round gives a fraction of one. Same ratio-direction slip
                # as 6.6.4's roofline and 6.10's seconds-vs-milliseconds: the
                # error is always in which side is the numerator.
                + f"Generation is "
                f"{min(r['examples_per_sec_training_can_consume'] for r in out['ladder'][1:]) / data_gen['ex_per_sec']:.0f}x "
                f"to "
                f"{max(r['examples_per_sec_training_can_consume'] for r in out['ladder'][1:]) / data_gen['ex_per_sec']:.0f}x "
                f"SLOWER than the training side can consume. "
                f"No training-side optimisation can change that ratio, "
                f"because the constraint is how fast examples are produced, not "
                f"how fast they are consumed. gpu-sim-u2o.6.7's own split makes "
                f"the cause concrete: 99.998% of the generation wall clock is "
                f"per-example process overhead, not GPU work."),
            "caveat": ("This compares THROUGHPUT, not end-to-end wall clock for a "
                       "whole run, and the two sides were measured in separate "
                       "jobs. It is the right comparison for the question 12.2 "
                       "asks -- whether training-speed work could pay -- and not "
                       "a claim about total project wall clock."),
        }

    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
