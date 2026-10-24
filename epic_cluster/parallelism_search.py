"""5D parallelism search (gpu-sim-5o4.1).

Enumerates (tp, pp, cp, dp, zero) with tp*pp*cp*dp == N, scores each with
epic_cluster.nccl_cost, reports sum (no overlap) and max (full overlap) step
time, memory feasibility, and dominant term. Ranks by SUM.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from epic_cluster import nccl_cost
try:
    from epic_cluster import link_model
except Exception:
    link_model = None

# Llama-3-8B shape (verified against public Llama-3 spec: 32 layers, hidden 4096,
# intermediate 14336, 32 heads, head_dim 128, vocab 128256, context 8192).
MODEL = {
    "n_layers": 32,
    "d_model": 4096,
    "n_heads": 32,
    "head_dim": 128,
    "d_ff": 14336,
    "vocab": 128256,
    "seq": 8192,
}


def total_params(m):
    d = m["d_model"]; ff = m["d_ff"]; V = m["vocab"]; L = m["n_layers"]
    per_layer = 4 * d * d + 3 * d * ff + 2 * d
    emb = V * d
    return L * per_layer + emb + d  # final norm


@dataclass
class Config:
    tp: int; pp: int; cp: int; dp: int; zero: int
    step_sum: float
    step_max: float
    compute: float
    tp_comm: float
    cp_comm: float
    pp_comm: float
    dp_comm: float
    memory_gb: float
    feasible: bool
    reason: str
    dominant: str


def divisors_product(N):
    out = []
    for tp in range(1, N + 1):
        if N % tp: continue
        for pp in range(1, N // tp + 1):
            if (N // tp) % pp: continue
            for cp in range(1, N // (tp * pp) + 1):
                if (N // (tp * pp)) % cp: continue
                dp = N // (tp * pp * cp)
                out.append((tp, pp, cp, dp))
    return out


def score(tp, pp, cp, dp, zero, hw, m):
    N = tp * pp * cp * dp
    P = total_params(m)
    seq = m["seq"]
    d = m["d_model"]
    dtype_bytes = 2  # bf16
    peak = hw["peak_flops"]  # per GPU
    mem_limit = hw["mem_gb"] * 1e9
    fabric = hw.get("fabric", "nvlink4_h100")
    spec_path = hw.get("spec_path")
    spec = None
    if spec_path:
        import json
        with open(spec_path) as _f:
            spec = json.load(_f)

    # params per gpu
    params_per_gpu = P / (tp * pp)
    if zero == 3 and dp > 1:
        params_per_gpu = params_per_gpu / dp
    # grads: ZeRO-2/3 shard grads across dp
    if zero >= 2 and dp > 1:
        grads_per_gpu = (P / (tp * pp)) / dp
    else:
        grads_per_gpu = P / (tp * pp)
    # opt state (8 bytes/param) sharded if zero>=1
    opt_per_gpu = (P / (tp * pp)) * 8
    if zero >= 1 and dp > 1:
        opt_per_gpu = opt_per_gpu / dp
    if zero == 3 and dp > 1:
        opt_per_gpu = (P / (tp * pp)) * 8 / dp
    # params/grads bytes
    param_bytes_per_gpu = params_per_gpu * dtype_bytes
    grad_bytes_per_gpu = grads_per_gpu * dtype_bytes

    # activations per gpu
    layers_per_stage = m["n_layers"] / pp
    seq_per_cp = seq / cp
    act_per_gpu = layers_per_stage * seq_per_cp * (d // tp) * dtype_bytes * 2  # fwd+bwd, batch 1, TP-sharded

    total_mem = param_bytes_per_gpu + grad_bytes_per_gpu + opt_per_gpu + act_per_gpu
    feasible = total_mem <= mem_limit
    reason = "" if feasible else f"mem {total_mem/1e9:.1f}GB > {mem_limit/1e9:.1f}GB"

    # compute
    tokens_per_gpu = seq / (cp * dp)  # batch 1 sequence
    flops_per_gpu = 6 * (P / (tp * pp)) * tokens_per_gpu
    compute = flops_per_gpu / peak
    # PP bubble (GPipe): bubble = (pp-1)/(micro_batches+pp-1) of compute
    if pp > 1:
        micro_batches = pp
        bubble = compute * (pp - 1) / (micro_batches + pp - 1)
        compute = compute + bubble

    # TP comm
    tp_comm = 0.0
    if tp > 1:
        msg_count = d * seq_per_cp
        if spec is not None:
            group = [((0 * tp + t) * cp + 0) * dp + 0 for t in range(tp)]
            pred = link_model.predict(spec, "ring_allreduce", msg_count, "float32", ranks=group, pieces_per_phase=1)
            tp_comm = m["n_layers"] * 4 * pred["duration_s"]
        else:
            msg = d * seq_per_cp * dtype_bytes
            pred = nccl_cost.predict("allreduce", tp, msg, fabric=fabric)
            tp_comm = m["n_layers"] * 4 * pred.total_s

    # CP comm
    cp_comm = 0.0
    if cp > 1:
        msg_bytes = seq_per_cp * d * dtype_bytes
        if spec is not None:
            group = [((0 * tp + 0) * cp + c) * dp + 0 for c in range(cp)]
            total = 0.0
            for i in range(cp - 1):
                total += link_model.point_to_point(spec, group[i], group[i+1], msg_bytes)
            cp_comm = m["n_layers"] * 2 * total
        else:
            msg = seq_per_cp * d * dtype_bytes
            pred = nccl_cost.ring_exchange(msg, hops=cp - 1, fabric=fabric)
            cp_comm = m["n_layers"] * 2 * pred.total_s

    # PP comm
    pp_comm = 0.0
    if pp > 1:
        microbatches = max(pp, 1)
        act_size = seq * d * dtype_bytes
        if spec is not None:
            def stage_rank(i):
                return (((i * tp + 0) * cp + 0) * dp + 0)
            boundary_cost = 0.0
            for i in range(pp - 1):
                boundary_cost += link_model.point_to_point(spec, stage_rank(i), stage_rank(i+1), act_size)
            pp_comm = microbatches * boundary_cost
        else:
            pred = nccl_cost.predict("sendrecv", 2, act_size, fabric=fabric)
            pp_comm = microbatches * 2 * (pp - 1) * pred.total_s

    # DP/FSDP comm
    dp_comm = 0.0
    if dp > 1:
        if spec is not None:
            if zero <= 1:
                group = [((0 * tp + 0) * cp + 0) * dp + d for d in range(dp)]
                pred = link_model.predict(spec, "ring_allreduce", int(grad_bytes_per_gpu // dtype_bytes), "float32", ranks=group, pieces_per_phase=1)
                dp_comm = pred["duration_s"]
            else:
                group = [((0 * tp + 0) * cp + 0) * dp + d for d in range(dp)]
                p1 = link_model.predict(spec, "ring_reduce_scatter", int(grad_bytes_per_gpu // dtype_bytes), "float32", ranks=group, pieces_per_phase=1)
                p2 = link_model.predict(spec, "ring_allgather", int(param_bytes_per_gpu // dtype_bytes), "float32", ranks=group, pieces_per_phase=1)
                dp_comm = p1["duration_s"] + p2["duration_s"]
        else:
            if zero <= 1:
                pred = nccl_cost.predict("allreduce", dp, int(grad_bytes_per_gpu), fabric=fabric)
                dp_comm = pred.total_s
            else:
                pred1 = nccl_cost.predict("reducescatter", dp, int(grad_bytes_per_gpu), fabric=fabric)
                pred2 = nccl_cost.predict("allgather", dp, int(param_bytes_per_gpu), fabric=fabric)
                dp_comm = pred1.total_s + pred2.total_s

    comms = tp_comm + cp_comm + pp_comm + dp_comm
    step_sum = compute + comms
    step_max = max(compute, comms)

    terms = {"compute": compute, "tp": tp_comm, "cp": cp_comm, "pp": pp_comm, "dp": dp_comm}
    dominant = max(terms, key=lambda k: terms[k])

    return Config(tp, pp, cp, dp, zero, step_sum, step_max, compute, tp_comm, cp_comm, pp_comm, dp_comm,
                  total_mem / 1e9, feasible, reason, dominant)


def search(N, hw, m=MODEL):
    results = []
    for tp, pp, cp, dp in divisors_product(N):
        zeros = [0] if dp == 1 else [0, 1, 2, 3]
        for zero in zeros:
            cfg = score(tp, pp, cp, dp, zero, hw, m)
            results.append(cfg)
    feasible = [c for c in results if c.feasible]
    feasible.sort(key=lambda c: c.step_sum)
    return results, feasible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gpus", type=int, default=8)
    ap.add_argument("--fabric", default="nvlink4_h100")
    ap.add_argument("--peak-tflops", type=float, default=989.0)  # H100 bf16
    ap.add_argument("--mem-gb", type=float, default=80.0)
    ap.add_argument("--tokens", type=int, default=None)
    args = ap.parse_args()

    hw = {"peak_flops": args.peak_tflops * 1e12, "mem_gb": args.mem_gb, "fabric": args.fabric}
    results, feasible = search(args.n_gpus, hw)

    print(f"# 5D search: N={args.n_gpus} fabric={args.fabric} peak={args.peak_tflops}TF mem={args.mem_gb}GB")
    print(f"total configs: {len(results)} feasible: {len(feasible)}")
    print()
    print("| tp | pp | cp | dp | zero | step_ms_sum | step_ms_max | dominant | mem_GB | feasible |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in feasible[:10]:
        print(f"| {c.tp} | {c.pp} | {c.cp} | {c.dp} | {c.zero} | {c.step_sum*1000:.2f} | {c.step_max*1000:.2f} | {c.dominant} | {c.memory_gb:.1f} | yes |")
    if feasible:
        w = feasible[0]
        print()
        print(f"WINNER: tp={w.tp} pp={w.pp} cp={w.cp} dp={w.dp} zero={w.zero} "
              f"step={w.step_sum*1000:.2f}ms dominant={w.dominant}")
    # excluded sample
    infeas = [c for c in results if not c.feasible][:5]
    if infeas:
        print()
        print("Excluded (sample):")
        for c in infeas:
            print(f"  tp={c.tp} pp={c.pp} cp={c.cp} dp={c.dp} zero={c.zero}: {c.reason}")


if __name__ == "__main__":
    main()
