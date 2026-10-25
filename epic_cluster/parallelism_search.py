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
    cp_grad: float
    pp_comm: float
    dp_comm: float
    memory_gb: float = 0.0
    feasible: bool = True
    reason: str = ""
    dominant: str = ""
    link_kinds: dict = None
    mem_params_gb: float = 0.0
    mem_grads_gb: float = 0.0
    mem_master_gb: float = 0.0
    mem_opt_gb: float = 0.0
    mem_act_gb: float = 0.0
    mem_attn_ws_gb: float = 0.0
    mem_logits_gb: float = 0.0
    mem_cp_buf_gb: float = 0.0


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
    # 5o4.5 sensitivity: allow the legacy single-fabric path to be swept over
    # alpha/bw without touching the module table. Ignored when spec_path is set
    # (the topology path reads its parameters from the spec).
    _alpha_override = hw.get("alpha_s")
    _bw_override = hw.get("bw_gbps")
    if spec_path is None and (_alpha_override is not None or _bw_override is not None):
        base = nccl_cost.FABRICS.get(fabric, {"alpha_s": 1.5e-6, "bw_gbps": 450.0})
        ov = dict(base)
        if _alpha_override is not None:
            ov["alpha_s"] = _alpha_override
        if _bw_override is not None:
            ov["bw_gbps"] = _bw_override
        nccl_cost.FABRICS["__ov__"] = ov
        fabric = "__ov__"
    spec = None
    if spec_path:
        import json
        with open(spec_path) as _f:
            spec = json.load(_f)

    # ---- Per-rank memory model (gpu-sim-6y1) ----
    # Assumptions stated per term. Mixed precision (bf16/fp16) is assumed:
    # params/grads/activations in dtype_bytes, master weights + optimizer
    # states in fp32. flash_attn (default True) avoids O(S^2) score storage.
    mixed_precision = hw.get("mixed_precision", True)
    flash_attn = hw.get("flash_attn", True)
    # recompute knob reduces activation term (gpu-sim-ooj): none=store all,
    # selective=recompute attention (store residuals only), full=recompute all.
    recompute = hw.get("recompute", "none")
    # gpu-sim-qk9: DP/FSDP comm audit knobs
    dp_dtype_str = "bfloat16" if hw.get("dp_dtype", "bf16") == "bf16" else "float32"
    dp_count_mode = hw.get("dp_count_mode", "flat")  # "flat" or "per_param"
    # estimated number of parameter tensors per rank (for per_param latency)
    dp_param_tensors = hw.get("dp_param_tensors", m["n_layers"] * 9 + 2)

    # params per gpu: ZeRO-3 shards params across dp
    params_per_gpu = P / (tp * pp)
    if zero == 3 and dp > 1:
        params_per_gpu = params_per_gpu / dp
    # grads: ZeRO-2/3 shard grads across dp
    if zero >= 2 and dp > 1:
        grads_per_gpu = (P / (tp * pp)) / dp
    else:
        grads_per_gpu = P / (tp * pp)
    # fp32 master weights: ZeRO-1 shards optimizer state (incl. master) across dp
    if mixed_precision:
        master_per_gpu = (P / (tp * pp))
        if zero >= 1 and dp > 1:
            master_per_gpu = master_per_gpu / dp
    else:
        master_per_gpu = 0.0
    # optimizer state m+v in fp32 (8 B/param), ZeRO-1 shards across dp
    opt_per_gpu = (P / (tp * pp)) * 8
    if zero >= 1 and dp > 1:
        opt_per_gpu = opt_per_gpu / dp

    param_bytes = params_per_gpu * dtype_bytes
    grad_bytes = grads_per_gpu * dtype_bytes
    master_bytes = master_per_gpu * 4.0
    opt_bytes = opt_per_gpu  # already in bytes (8 B/param)

    # activations: per-tensor widths (gpu-sim-6y1 rework).
    # A transformer layer stores, for backward, four tensors per token:
    #   input/residual (d), attention output (d), ffn intermediate (d_ff),
    #   ffn output (d).  Each is TP-sharded where the producing linear is
    #   sharded (all of them are, under standard TP), so widths are d/tp and
    #   d_ff/tp.  Previously this used 4*d, undercounting the FFN term by
    #   ~1.63x (d_ff=14336 vs 3*d=12288 for Llama-3 8B).
    # act_factor: backward reuses the stored forward activations; gradients
    #   live in grads_bytes, not here.  torch validation on a 5090 shows the
    #   measured activation delta is ~0.5x the S^2 score-storage term (torch
    #   frees scores during backward), so the S^2 term is an UPPER BOUND and
    #   the activation term is UNVALIDATED for exact fit; feasibility verdicts
    #   at 8B/70B scale are dominated by params/master/opt, not activation.
    layers_per_stage = m["n_layers"] / pp
    seq_per_cp = seq / cp
    d_tp = d // tp
    d_ff_tp = m["d_ff"] // tp
    # per-tensor widths: residual(d) + attn_out(d) + ffn_intermediate(d_ff) + ffn_out(d)
    per_token_width = 3 * d_tp + d_ff_tp
    if recompute == "none":
        act_factor = 2.3   # calibrated (gpu-sim-4nf): instrumented per-layer peak shows ~2.3x the per-tensor-width estimate (matmul intermediates, gradients in backward)
        layers_act = layers_per_stage
    elif recompute == "selective":
        act_factor = 0.5   # residuals only; attention recomputed
        layers_act = layers_per_stage
    else:  # full
        act_factor = 1.0
        layers_act = 1.0   # only one layer's activations live at a time
    # gpu-sim-54m: allow hw to override the activation factor (default: recompute-based)
    act_factor = hw.get("act_factor", act_factor)
    act_bytes = layers_act * seq_per_cp * dtype_bytes * per_token_width * act_factor

    # attention workspace (scores): [heads/tp, seq/cp, seq] if not flash
    if flash_attn:
        attn_ws_bytes = 0.0
    else:
        attn_ws_bytes = layers_per_stage * (m["n_heads"] // tp) * seq_per_cp * seq * 4.0

    # logits: [seq/cp, V/tp] (LM head sharded by tp, seq by cp)
    logits_bytes = (seq / cp) * (m["vocab"] // tp) * dtype_bytes

    # CP ring buffers: (cp-1) in-flight K/V blocks of (seq/cp)*(d/tp)
    if cp > 1:
        cp_buf_bytes = 2.0 * (cp - 1) * seq_per_cp * d_tp * dtype_bytes
    else:
        cp_buf_bytes = 0.0

    total_mem_bytes = (param_bytes + grad_bytes + master_bytes + opt_bytes
                       + act_bytes + attn_ws_bytes + logits_bytes + cp_buf_bytes)
    total_mem_gb = total_mem_bytes / 1e9

    feasible = total_mem_bytes <= mem_limit
    reason = "" if feasible else f"mem {total_mem_bytes/1e9:.1f}GB > {mem_limit/1e9:.1f}GB"

    # compute
    tokens_per_gpu = seq / (cp * dp)  # batch 1 sequence
    flops_per_gpu = 6 * (P / (tp * pp)) * tokens_per_gpu
    # attention FLOPs (fwd+bwd): 12 * L * S^2 * d / (tp*pp*cp*dp)
    # Derivation: per layer, per rank: local queries (S/cp) x all keys (S) x
    # (d/tp) heads, for QK^T and AV = 4 * (S/cp) * S * (d/tp) MACs fwd;
    # bwd recomputes scores (3x) => 12 * S^2 * d / (cp*tp) per layer for one
    # sample; sharded across pp stages (layers/pp) and dp ranks (batch slice)
    # gives /pp and /dp. Equivalently (12*L*S*d)*S/N with N=tp*pp*cp*dp.
    # Ring assumption: local queries attend all keys (gathered via ring).
    attn_flops = 12.0 * m["n_layers"] * (seq ** 2) * d / (tp * pp * cp * dp)
    flops_per_gpu = flops_per_gpu + attn_flops
    # recompute knob
    recompute = hw.get("recompute", "none")
    if recompute == "full":
        flops_per_gpu = flops_per_gpu + (6 * (P / (tp * pp)) * tokens_per_gpu) + attn_flops
    elif recompute == "selective":
        flops_per_gpu = flops_per_gpu + attn_flops
    compute = flops_per_gpu / peak
    # PP bubble (GPipe): idle time, NOT FLOPs. Kept out of `compute` so the
    # compute term stays invariant across pp splits (gpu-sim-i95 conservation).
    bubble = 0.0
    if pp > 1:
        micro_batches = hw.get("micro_batches", pp)
        bubble = compute * (pp - 1) / (micro_batches + pp - 1)

    # TP comm
    tp_comm = 0.0
    tp_kind = None
    if tp > 1:
        msg_count = d * seq_per_cp
        if spec is not None:
            group = [((0 * tp + t) * cp + 0) * dp + 0 for t in range(tp)]
            pred = link_model.predict(spec, "ring_allreduce", msg_count, "bfloat16", ranks=group, pieces_per_phase=1)
            tp_comm = m["n_layers"] * 4 * pred["duration_s"]
            tp_kind = pred["bottleneck"]["kind"]
        else:
            msg = d * seq_per_cp * dtype_bytes
            pred = nccl_cost.predict("allreduce", tp, msg, fabric=fabric)
            tp_comm = m["n_layers"] * 4 * pred.total_s

    # CP comm
    cp_comm = 0.0
    cp_kind = None
    if cp > 1:
        msg_bytes = seq_per_cp * d * dtype_bytes
        if spec is not None:
            group = [((0 * tp + 0) * cp + c) * dp + 0 for c in range(cp)]
            total = 0.0
            cp_kind = None
            for i in range(cp - 1):
                dur, kind = link_model.point_to_point(spec, group[i], group[i+1], msg_bytes)
                total += dur
                cp_kind = kind
            cp_comm = m["n_layers"] * 2 * total
        else:
            msg = seq_per_cp * d * dtype_bytes
            pred = nccl_cost.ring_exchange(msg, hops=cp - 1, fabric=fabric)
            cp_comm = m["n_layers"] * 2 * pred.total_s

    # CP gradient sync (gpu-sim-olg): 8jg added dist.all_reduce(prm.grad, SUM, cp_group)
    # then div_(cp) per parameter. Crude: full all-reduce of P/(tp*pp) params over cp
    # ranks. Production alternative: reduce-scatter the sequence-sharded shards
    # (~2*(cp-1)/cp * bytes/cp). Harness does crude.
    cp_grad = 0.0
    cp_grad_kind = None  # only populated on the spec path (link_model); nccl path has no .kind
    cp_grad_mode = hw.get("cp_grad_mode", "crude")
    if cp > 1:
        grad_bytes = (P / (tp * pp)) * dtype_bytes
        if cp_grad_mode == "production":
            msg_bytes = grad_bytes / cp
            algo = "reducescatter"
        else:
            msg_bytes = grad_bytes
            algo = "allreduce"
        if spec is not None:
            group = [((0 * tp + 0) * cp + c) * dp + 0 for c in range(cp)]
            if algo == "allreduce":
                pred = link_model.predict(spec, "ring_allreduce", int(msg_bytes // dtype_bytes), dp_dtype_str, ranks=group, pieces_per_phase=1)
                cp_grad = pred["duration_s"]
                cp_grad_kind = pred["bottleneck"]["kind"]
            else:
                pred = link_model.predict(spec, "ring_reduce_scatter", int(msg_bytes // dtype_bytes), dp_dtype_str, ranks=group, pieces_per_phase=1)
                cp_grad = pred["duration_s"]
                cp_grad_kind = pred["bottleneck"]["kind"]
        else:
            if algo == "allreduce":
                pred = nccl_cost.predict("allreduce", cp, msg_bytes, fabric=fabric)
            else:
                pred = nccl_cost.predict("reducescatter", cp, msg_bytes, fabric=fabric)
            cp_grad = pred.total_s
            cp_grad_kind = getattr(pred, "kind", None)

    # PP comm
    pp_comm = 0.0
    pp_kind = None
    if pp > 1:
        microbatches = max(pp, 1)
        act_size = seq * d * dtype_bytes
        if spec is not None:
            def stage_rank(i):
                return (((i * tp + 0) * cp + 0) * dp + 0)
            boundary_cost = 0.0
            pp_kind = None
            for i in range(pp - 1):
                dur, kind = link_model.point_to_point(spec, stage_rank(i), stage_rank(i+1), act_size)
                boundary_cost += dur
                pp_kind = kind
            pp_comm = microbatches * boundary_cost
        else:
            pred = nccl_cost.predict("sendrecv", 2, act_size, fabric=fabric)
            pp_comm = microbatches * 2 * (pp - 1) * pred.total_s

    # DP/FSDP comm (gpu-sim-qk9: audited)
    # Named scheme: production ZeRO semantics:
    #   zero<=1 (DDP/ZeRO-1): all_reduce of grads only (grads stay full; params stay full)
    #   zero==2 (ZeRO-2):      reduce_scatter of grads only (grads sharded; params stay full)
    #   zero>=3 (ZeRO-3/FSDP): reduce_scatter of grads + all_gather of params
    # Payload dtype: bf16 (dtype_bytes) by default; fp32 only if implementation upcasts.
    # Count: "flat" (one collective per op, production/bucketed) by default.
    #        "per_param" multiplies latency by ~N_param_tensors (harness upper bound).
    dp_comm = 0.0
    dp_kind = None
    if dp > 1:
        if spec is not None:
            if zero <= 1:
                group = [((0 * tp + 0) * cp + 0) * dp + d for d in range(dp)]
                pred = link_model.predict(spec, "ring_allreduce", int(grads_per_gpu), dp_dtype_str, ranks=group, pieces_per_phase=1)
                dp_comm = pred["duration_s"]
                dp_kind = pred["bottleneck"]["kind"]
            elif zero == 2:
                group = [((0 * tp + 0) * cp + 0) * dp + d for d in range(dp)]
                pred = link_model.predict(spec, "ring_reduce_scatter", int(grads_per_gpu), dp_dtype_str, ranks=group, pieces_per_phase=1)
                dp_comm = pred["duration_s"]
                dp_kind = pred["bottleneck"]["kind"]
            else:
                group = [((0 * tp + 0) * cp + 0) * dp + d for d in range(dp)]
                p1 = link_model.predict(spec, "ring_reduce_scatter", int(grads_per_gpu), dp_dtype_str, ranks=group, pieces_per_phase=1)
                p2 = link_model.predict(spec, "ring_allgather", int(params_per_gpu), dp_dtype_str, ranks=group, pieces_per_phase=1)
                dp_comm = p1["duration_s"] + p2["duration_s"]
                dp_kind = p1["bottleneck"]["kind"]
        else:
            _dpb = 2 if dp_dtype_str == "float32" else 1
            if zero <= 1:
                pred = nccl_cost.predict("allreduce", dp, int(grad_bytes * _dpb), fabric=fabric)
                dp_comm = pred.total_s
            elif zero == 2:
                pred = nccl_cost.predict("reducescatter", dp, int(grad_bytes * _dpb), fabric=fabric)
                dp_comm = pred.total_s
            else:
                pred1 = nccl_cost.predict("reducescatter", dp, int(grad_bytes * _dpb), fabric=fabric)
                pred2 = nccl_cost.predict("allgather", dp, int(param_bytes * _dpb), fabric=fabric)
                dp_comm = pred1.total_s + pred2.total_s

    if dp_count_mode == "per_param" and dp_comm > 0:
        dp_comm = dp_comm * dp_param_tensors
    comms = tp_comm + cp_comm + cp_grad + pp_comm + dp_comm
    step_compute = compute + bubble
    step_sum = step_compute + comms
    step_max = max(step_compute, comms)

    terms = {"compute": step_compute, "tp": tp_comm, "cp": cp_comm, "cp_grad": cp_grad, "pp": pp_comm, "dp": dp_comm}
    dominant = max(terms, key=lambda k: terms[k])

    kinds = {"tp": tp_kind, "cp": cp_kind, "pp": pp_kind, "dp": dp_kind}
    if spec is None:
        kinds = {"tp": fabric, "cp": fabric, "pp": fabric, "dp": fabric}
    return Config(tp, pp, cp, dp, zero, step_sum, step_max, compute, tp_comm, cp_comm, cp_grad, pp_comm, dp_comm,
                  total_mem_bytes / 1e9, feasible, reason, dominant, kinds,
                  param_bytes/1e9, grad_bytes/1e9, master_bytes/1e9, opt_bytes/1e9,
                  act_bytes/1e9, attn_ws_bytes/1e9, logits_bytes/1e9, cp_buf_bytes/1e9)


def search(N, hw, m=MODEL, rank_by="sum"):
    results = []
    for tp, pp, cp, dp in divisors_product(N):
        zeros = [0] if dp == 1 else [0, 1, 2, 3]
        for zero in zeros:
            cfg = score(tp, pp, cp, dp, zero, hw, m)
            results.append(cfg)
    feasible = [c for c in results if c.feasible]
    key = (lambda c: c.step_sum) if rank_by == "sum" else (lambda c: c.step_max)
    feasible.sort(key=key)
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
