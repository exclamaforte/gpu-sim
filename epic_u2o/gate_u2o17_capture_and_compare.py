#!/usr/bin/env python3
"""u2o.17.2 / u2o.17.4: capture a real RTX 5090 torch.profiler trace for a small
fixed PyTorch workload, compute a naive roofline-based "simulated" prediction
for the same ops (NOT the trained transformer -- that model does not exist yet,
blocked on epics .6/.9), and report the actual relative error between them.

This is an honest first pass at the definition-of-done gate (epic gpu-sim-u2o.17):
it does not use libfakegpu.so at all (see gpu-sim-u2o.16 -- LD_PRELOAD over a
real-CUDA-linked PyTorch process is an unsupported ABI-mismatched combination),
and it does not claim parity. It exists to (a) produce a committed real-trace
fixture, and (b) put a first real number on how wrong a flat/naive latency
model is, which is exactly the gap epics .6/.7/.9 exist to close.

RTX 5090 peak-FLOPs/bandwidth constants below are manufacturer-stated marketing
figures, NOT independently verified in this repo -- they are only used to
produce an illustrative naive baseline, not a calibrated prediction.
"""
import json
import time

import torch

RTX_5090_FP32_TFLOPS = 104.8e12   # marketing CUDA-core FP32 peak, unverified here
RTX_5090_BW_BYTES_S = 1568e9      # marketing GDDR7 bandwidth, unverified here

WARMUP = 5
TRIALS = 20


def real_trace():
    torch.cuda.init()
    x = torch.randn(4096, 4096, device="cuda")
    w = torch.randn(4096, 4096, device="cuda")
    for _ in range(WARMUP):
        y = x @ w
        torch.relu(y)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(TRIALS):
            y = x @ w
            torch.relu(y)
        torch.cuda.synchronize()

    # aten::relu itself dispatches to aten::clamp_min (its self time is ~0,
    # exactly like aten::matmul dispatches to aten::mm) -- use the leaf op
    # that actually carries the self-CUDA time, same pattern as matmul/mm.
    per_op = {}
    for e in prof.key_averages():
        key = {"aten::mm": "aten::mm", "aten::clamp_min": "aten::relu"}.get(e.key)
        if key:
            per_op[key] = {
                "self_cuda_time_total_us": e.self_device_time_total,
                "count": e.count,
                "self_cuda_time_avg_us": e.self_device_time_total / e.count,
            }
    return per_op


def naive_roofline_prediction():
    n = 4096
    matmul_flops = 2 * (n ** 3)
    matmul_pred_s = matmul_flops / RTX_5090_FP32_TFLOPS

    elems = n * n
    relu_bytes = 2 * elems * 4  # read + write, fp32
    relu_pred_s = relu_bytes / RTX_5090_BW_BYTES_S

    return {
        "aten::mm": {"predicted_time_avg_us": matmul_pred_s * 1e6},
        "aten::relu": {"predicted_time_avg_us": relu_pred_s * 1e6},
    }


def compare(real, predicted):
    report = {}
    for op in ("aten::mm", "aten::relu"):
        actual = real[op]["self_cuda_time_avg_us"]
        pred = predicted[op]["predicted_time_avg_us"]
        rel_err = (pred - actual) / actual
        report[op] = {
            "real_us": actual,
            "naive_predicted_us": pred,
            "relative_error": rel_err,
        }
    real_ratio = real["aten::mm"]["self_cuda_time_avg_us"] / real["aten::relu"]["self_cuda_time_avg_us"]
    pred_ratio = predicted["aten::mm"]["predicted_time_avg_us"] / predicted["aten::relu"]["predicted_time_avg_us"]
    report["rank_order"] = {
        "real_mm_over_relu_ratio": real_ratio,
        "naive_predicted_mm_over_relu_ratio": pred_ratio,
        "rank_order_matches": real_ratio > 1 and pred_ratio > 1,  # mm should dominate in both
    }
    return report


if __name__ == "__main__":
    real = real_trace()
    predicted = naive_roofline_prediction()
    report = compare(real, predicted)

    mm_err = report["aten::mm"]["relative_error"] * 100
    relu_err = report["aten::relu"]["relative_error"] * 100
    real_ratio = report["rank_order"]["real_mm_over_relu_ratio"]
    pred_ratio = report["rank_order"]["naive_predicted_mm_over_relu_ratio"]
    out = {
        "workload": "4096x4096 fp32 matmul + relu on the same tensor, 5 warmup + 20 trials",
        "device": torch.cuda.get_device_name(0),
        "real_measured": real,
        "naive_roofline_predicted": predicted,
        "comparison": report,
        "verdict": (
            f"FAIL -- naive flat roofline model is {mm_err:+.1f}% off on matmul and {relu_err:+.1f}% "
            f"off on relu, real mm/relu ratio {real_ratio:.1f}x vs predicted {pred_ratio:.1f}x this run "
            "(this ratio itself varies run to run -- see notes on measurement variance). "
            "Per-op-class predictions from a real trained model (epics .6/.7/.9, not yet built) are "
            "required for parity. Rank-order (mm dominates relu) does hold even with this naive model."
        ),
    }
    with open("epic_u2o/gate_u2o17_real_vs_naive_trace.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
