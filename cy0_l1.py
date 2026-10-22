"""cy0-L1: H100 rack prediction — one decoder layer, end to end."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from epic_cluster.spec import load as load_spec
from epic_cluster.link_model import predict as predict_collective

# ---------------------------------------------------------------------------
# Parameters (efficiency factors are PARAMETERS with provenance, not structure)
# ---------------------------------------------------------------------------

# Per-op-class compute efficiency.  Starting assumption: published MFU ranges
# for H100 dense transformer training are 40-55% for GEMM-class ops and
# lower for memory-bound ops.  Source: NVIDIA H100 whitepaper + Megatron-LM
# profiling (Shoeybi et al. 2019) typical MFU ~0.45-0.55 for large GEMMs.
EFF_CLASS: Dict[str, float] = {
    "gemm": 0.50,
    "attention": 0.35,
    "layernorm": 0.20,
    "elementwise": 0.30,
}
EFF_PROVENANCE = "published-spec assumption: H100 dense-transformer MFU ~0.45-0.55 for GEMM, lower for memory-bound ops (NVIDIA H100 whitepaper; Megatron-LM profiling). Refit from real traces in L4."

HIDDEN = 4096
N_HEADS = 32
HEAD_DIM = 128
FFN_DIM = 11008  # LLaMA-style ~2.7x hidden
SEQ_LEN = 2048
BATCH = 1
DTYPE = "bfloat16"
DTYPE_BYTES = 2
TP = 2  # tensor parallelism degree

# ---------------------------------------------------------------------------
# Work model: one dense transformer decoder layer
# ---------------------------------------------------------------------------


@dataclass
class Op:
    name: str
    kind: str  # "compute" or "comm"
    flops: int
    bytes: int
    op_class: str  # for compute: gemm/attention/layernorm/elementwise
    dtype: str = DTYPE
    provenance: str = ""


def layer_ops(hidden=HIDDEN, ffn=FFN_DIM, seq=SEQ_LEN, batch=BATCH,
              n_heads=N_HEADS, head_dim=HEAD_DIM) -> List[Op]:
    """Analytic work model for one dense transformer decoder layer."""
    B = batch
    S = seq
    H = hidden
    F = ffn
    ops: List[Op] = []

    # 1. QKV projection: 3 * H * H * (B*S) * 2 FLOPs (matmul)
    qkv_flops = 3 * H * H * B * S * 2
    qkv_bytes = 3 * H * H * DTYPE_BYTES + B * S * H * DTYPE_BYTES  # weights + input
    ops.append(Op("qkv_proj", "compute", qkv_flops, qkv_bytes, "gemm",
                  provenance="analytic: 3*H*H*B*S*2 FLOPs for Q,K,V projections"))

    # 2. Attention score: Q @ K^T -> (B, n_heads, S, S)
    attn_score_flops = B * n_heads * S * S * head_dim * 2
    attn_score_bytes = B * S * H * DTYPE_BYTES * 2  # Q and K
    ops.append(Op("attn_score", "compute", attn_score_flops, attn_score_bytes, "attention",
                  provenance="analytic: B*n_heads*S*S*head_dim*2 for Q@K^T"))

    # 3. Attention value: softmax(QK^T) @ V -> (B, n_heads, S, head_dim)
    attn_val_flops = B * n_heads * S * S * head_dim * 2
    attn_val_bytes = B * n_heads * S * S * DTYPE_BYTES + B * S * H * DTYPE_BYTES
    ops.append(Op("attn_value", "compute", attn_val_flops, attn_val_bytes, "attention",
                  provenance="analytic: B*n_heads*S*S*head_dim*2 for softmax@V"))

    # 4. Output projection: H * H * (B*S) * 2
    oproj_flops = H * H * B * S * 2
    oproj_bytes = H * H * DTYPE_BYTES + B * S * H * DTYPE_BYTES
    ops.append(Op("output_proj", "compute", oproj_flops, oproj_bytes, "gemm",
                  provenance="analytic: H*H*B*S*2 for output projection"))

    # 5. MLP1 (gate/up): 2 * H * F * (B*S) * 2
    mlp1_flops = 2 * H * F * B * S * 2
    mlp1_bytes = 2 * H * F * DTYPE_BYTES + B * S * H * DTYPE_BYTES
    ops.append(Op("mlp1", "compute", mlp1_flops, mlp1_bytes, "gemm",
                  provenance="analytic: 2*H*F*B*S*2 for gate+up projections"))

    # 6. MLP2 (down): F * H * (B*S) * 2
    mlp2_flops = F * H * B * S * 2
    mlp2_bytes = F * H * DTYPE_BYTES + B * S * F * DTYPE_BYTES
    ops.append(Op("mlp2", "compute", mlp2_flops, mlp2_bytes, "gemm",
                  provenance="analytic: F*H*B*S*2 for down projection"))

    # 7. Two layernorms: 2 * (B*S*H) * 5 FLOPs (mean, var, normalize)
    ln_flops = 2 * B * S * H * 5
    ln_bytes = 2 * B * S * H * DTYPE_BYTES
    ops.append(Op("layernorms", "compute", ln_flops, ln_bytes, "layernorm",
                  provenance="analytic: 2*B*S*H*5 for two layernorms"))

    # 8. TP allreduce for row/column parallel layers (comm)
    # Size: hidden_dim * batch * seq * dtype bytes
    ar_bytes = B * S * H * DTYPE_BYTES
    ops.append(Op("tp_allreduce", "comm", 0, ar_bytes, "comm",
                  provenance="analytic: B*S*H*dtype for TP allreduce payload"))

    return ops


def compute_time(op: Op, peak_flops: float, achieved_bw_gbs: float) -> float:
    """Roofline: t = max(FLOPs/(peak*eff), bytes/achieved_bw)."""
    eff = EFF_CLASS.get(op.op_class, 0.5)
    if op.flops == 0:
        compute_t = 0.0
    else:
        compute_t = op.flops / (peak_flops * eff)
    if op.bytes == 0:
        mem_t = 0.0
    else:
        mem_t = op.bytes / (achieved_bw_gbs * 1e9)
    return max(compute_t, mem_t)


def peak_flops(spec) -> float:
    """Derive peak FLOPs from spec: sm_count * clock_mhz * 2 (FMA) * ... (bf16 tensor core)."""
    model = list(spec["models"].values())[0]
    # H100: 132 SMs, 1830 MHz, 2 FMA per clock per SM (simplified), bf16 tensor cores
    # Use a simplified peak: sm_count * clock_mhz * 1e6 * 2 (FMA) * 2 (tensor) FLOPs/s
    # Actually, H100 peak bf16 dense = 989 TFLOPS. Derive from spec:
    sm = model["sm_count"]
    clock = model["clock_mhz"]
    # H100 tensor core: 4096 FLOPs per clock per SM for bf16 (approx)
    return sm * clock * 1e6 * 4096 / 1e12  # TFLOPS -> convert to FLOPS


def predict_layer(spec_path: str, tp: int = TP, hidden: int = HIDDEN,
                  ffn: int = FFN_DIM, seq: int = SEQ_LEN,
                  batch: int = BATCH) -> Dict[str, Any]:
    """Run the full L1 prediction, emit JSONL trace.

    Model shape is an EXPLICIT parameter (gpu-sim-cy4). It used to be read from
    module globals that callers monkey-patched, which cannot work: layer_ops
    binds its defaults at def time, so rebinding the global changed nothing and
    --seq/--batch were silently inert.
    """
    spec = load_spec(spec_path)
    pf = peak_flops(spec) * 1e12  # FLOPS
    model = list(spec["models"].values())[0]
    achieved_bw = model["mem_bw_gbs"]  # GB/s

    ops = layer_ops(hidden=hidden, ffn=ffn, seq=seq, batch=batch)

    # Resolve TP allreduce through link_model.predict()
    # Use first two ranks (0 and 1) as representative TP group
    ar_bytes = ops[-1].bytes  # tp_allreduce bytes
    # count = bytes / dtype_bytes
    ar_count = ar_bytes // 2  # bfloat16 = 2 bytes
    comm_result = predict_collective(spec, "ring_allreduce", ar_count, "bfloat16", n_ranks=tp, pieces_per_phase=8)
    comm_time = comm_result["duration_s"]

    records = []
    total_s = 0.0
    for op in ops:
        if op.kind == "compute":
            t = compute_time(op, pf, achieved_bw)
            rec = {
                "kind": "compute",
                "op": op.name,
                "bytes": op.bytes,
                "flops": op.flops,
                "seconds": t,
                "op_class": op.op_class,
                "efficiency": EFF_CLASS[op.op_class],
                "efficiency_provenance": EFF_PROVENANCE,
                "provenance": op.provenance,
                "predicted": True,
            }
        elif op.name == "tp_allreduce":
            rec = {
                "kind": "comm",
                "op": op.name,
                "bytes": op.bytes,
                "flops": 0,
                "seconds": comm_time,
                "topology_path": comm_result.get("bottleneck", {}),
                "provenance": op.provenance,
                "predicted": True,
            }
        else:
            rec = {
                "kind": op.kind,
                "op": op.name,
                "bytes": op.bytes,
                "flops": op.flops,
                "seconds": 0.0,
                "provenance": op.provenance,
                "predicted": True,
            }
        records.append(rec)
        total_s += rec["seconds"]

    return {
        "total_seconds": total_s,
        "components": records,
        "assumptions": [
            "eff_class is a published-spec assumption (H100 MFU ~0.45-0.55), not fitted",
            "TP allreduce uses link_model.predict() on the rack spec topology",
            "No PP bubbles, no FSDP, no overlap (those are L2+)",
            "predicted:true -- this is a prediction, not a measurement",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="specs/h100_rack.json")
    parser.add_argument("--tp", type=int, default=TP)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = predict_layer(args.spec, tp=args.tp)

    # Emit JSONL trace
    lines = []
    for rec in result["components"]:
        lines.append(json.dumps(rec))
    trace_text = "\n".join(lines) + "\n"

    if args.output:
        Path(args.output).write_text(trace_text)
    else:
        print(trace_text, end="")

    # Summary to stderr
    import sys
    print(f"total_seconds={result['total_seconds']:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
