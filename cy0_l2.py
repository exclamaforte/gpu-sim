"""cy0-L2: full training-step composer — fwd/bwd/optimizer, DP allreduce, PP bubbles, overlap policy."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import cy0_l1
from epic_cluster.spec import load as load_spec
from epic_cluster.link_model import predict as predict_collective

# ---------------------------------------------------------------------------
# Overlap policy: NAMED, CONFIGURABLE assumption with provenance
# ---------------------------------------------------------------------------
OVERLAP_POLICY = {
    "tp_allreduce_hidden": True,   # TP allreduce hidden behind layer compute
    "dp_allreduce_exposed": True,  # DP gradient allreduce exposed after backward
    "pp_sends_exposed": True,      # PP microbatch sends exposed
}
OVERLAP_PROVENANCE = (
    "Assumption: TP allreduce (small, intra-node NVLink) is hidden behind "
    "layer compute; DP allreduce (large, cross-node IB) and PP sends are "
    "exposed. Based on Megatron-LM profiling (Shoeybi et al. 2019). "
    "Refit from real traces in L4."
)


@dataclass
class StepConfig:
    n_layers: int = 2
    n_microbatches: int = 4
    dp: int = 2   # data parallelism degree
    tp: int = 2   # tensor parallelism degree
    pp: int = 1   # pipeline parallelism degree
    overlap: bool = True
    # Model shape as EXPLICIT fields (gpu-sim-cy4). These used to be read from
    # cy0_l1 module globals that cy0_l3 monkey-patched, which had no effect:
    # layer_ops binds its defaults at def time, so --seq and --batch were inert.
    hidden: int = cy0_l1.HIDDEN
    ffn: int = cy0_l1.FFN_DIM
    seq: int = cy0_l1.SEQ_LEN
    batch: int = cy0_l1.BATCH


def layer_compute_time(spec_path: str, cfg: "StepConfig" = None) -> float:
    """Compute time for one layer (sum of compute ops, no comm).

    Model shape comes from cfg (gpu-sim-cy4). Defaults reproduce the old
    behaviour for callers that pass nothing.
    """
    kw = {}
    if cfg is not None:
        kw = {"hidden": cfg.hidden, "ffn": cfg.ffn, "seq": cfg.seq, "batch": cfg.batch}
    result = cy0_l1.predict_layer(spec_path, tp=1, **kw)
    return sum(c["seconds"] for c in result["components"] if c["kind"] == "compute")


def layer_tp_comm_time(spec_path: str, tp: int, cfg: "StepConfig" = None) -> float:
    """TP allreduce time for one layer.

    Two allreduces per layer: (1) attention output (row-parallel) and
    (2) MLP down (row-parallel). Each is a full B*S*H payload.
    tp=1 means no tensor parallelism: zero TP comm.
    """
    if tp <= 1:
        return 0.0
    kw = {}
    if cfg is not None:
        kw = {"hidden": cfg.hidden, "ffn": cfg.ffn, "seq": cfg.seq, "batch": cfg.batch}
    result = cy0_l1.predict_layer(spec_path, tp=tp, **kw)
    comm = [c for c in result["components"] if c["kind"] == "comm"]
    single = comm[0]["seconds"] if comm else 0.0
    return 2.0 * single


def dp_allreduce_time(spec_path: str, dp: int, grad_bytes: int) -> float:
    """DP gradient allreduce time (cross-node)."""
    if dp <= 1:
        return 0.0
    count = grad_bytes // 4  # float32
    res = predict_collective(
        load_spec(spec_path), "ring_allreduce", count, "float32",
        n_ranks=dp, pieces_per_phase=8,
    )
    return res["duration_s"]


def pp_bubble_time(n_microbatches: int, n_stages: int, n_layers: int,
                  layer_time: float) -> float:
    """Pipeline fill + drain, derived from the stage geometry (gpu-sim-1p9).

    A pp-way pipeline splits the model into n_stages stages, so each stage holds
    n_layers / n_stages layers and one microbatch occupies a stage for
        t_stage = (n_layers / n_stages) * layer_time
    Fill costs (n_stages - 1) stages of idle time and drain costs the same, so

        bubble = (n_stages - 1) * (n_layers / n_stages) * layer_time

    The previous version returned (n_stages - 1) * layer_time, which is the
    n_layers == n_stages case only. It was invisible because StepConfig defaults
    to pp=1, where both expressions are zero; at 8 layers and pp=2 it understated
    the bubble by 4x.

    n_microbatches does not enter an ABSOLUTE fill+drain cost: more microbatches
    shrink the bubble FRACTION, (pp-1)/(m+pp-1), but the absolute idle time is
    set by the stage count and the per-stage pass. It is accepted here so callers
    need not special-case it, and named as unused on purpose.
    """
    del n_microbatches  # absolute fill+drain is independent of m; see docstring
    if n_stages <= 1:
        return 0.0
    layers_per_stage = n_layers / n_stages
    return (n_stages - 1) * layers_per_stage * layer_time


def predict_step(spec_path: str, cfg: StepConfig) -> Dict[str, Any]:
    """Predict one training step."""
    spec = load_spec(spec_path)
    lc = layer_compute_time(spec_path, cfg)
    tp_comm = layer_tp_comm_time(spec_path, cfg.tp, cfg)

    # Per-layer forward time
    if cfg.overlap and OVERLAP_POLICY["tp_allreduce_hidden"]:
        # TP comm hidden behind compute, but only up to compute time.
        # Excess comm beyond compute is exposed (cannot hide more than compute).
        exposed_tp = max(0.0, tp_comm - lc)
        fwd_layer = lc + exposed_tp
        overlapped_tp = min(tp_comm, lc)
    else:
        fwd_layer = lc + tp_comm
        overlapped_tp = 0.0
        exposed_tp = tp_comm

    # Backward: same compute + TP comm (gradient)
    if cfg.overlap and OVERLAP_POLICY["tp_allreduce_hidden"]:
        exposed_tp_bwd = max(0.0, tp_comm - lc)
        bwd_layer = lc + exposed_tp_bwd
    else:
        bwd_layer = lc + tp_comm

    # Optimizer: Adam, roofline (memory-bound). 12 FLOP/param vs 28 B/param
    # (fp32_master: read+write {p,m,v,g}); ZeRO-1 shards across dp.
    H = cy0_l1.HIDDEN
    F = cy0_l1.FFN_DIM
    params_per_layer = 4 * H * H + 3 * H * F
    total_params = params_per_layer * cfg.n_layers
    params_per_gpu = total_params / max(1, cfg.dp)  # ZeRO-1 shard
    opt_flops = 12 * params_per_gpu
    bytes_per_param = 28  # fp32_master convention (scup vocabulary)
    opt_bytes = int(bytes_per_param * params_per_gpu)
    opt_op = cy0_l1.Op(name="optimizer", kind="compute", flops=opt_flops,
                       bytes=opt_bytes, op_class="optimizer",
                       provenance="analytic: Adam fp32_master 12 FLOP/param, 28 B/param (read+write p,m,v,g), ZeRO-1 shard across dp")
    spec = load_spec(spec_path)
    pf = cy0_l1.peak_flops(spec) * 1e12
    achieved_bw = list(spec["models"].values())[0]["mem_bw_gbs"]
    opt_time = cy0_l1.compute_time(opt_op, pf, achieved_bw)

    # DP allreduce: after backward, gradient sync
    # Gradient size = dtype_bytes (fp32=4) * params_per_layer * n_layers
    H = cy0_l1.HIDDEN
    F = cy0_l1.FFN_DIM
    params_per_layer = 4 * H * H + 3 * H * F
    grad_bytes = 4 * params_per_layer * cfg.n_layers  # fp32 gradients
    dp_time = dp_allreduce_time(spec_path, cfg.dp, grad_bytes)
    if cfg.overlap and OVERLAP_POLICY["dp_allreduce_exposed"]:
        dp_exposed = dp_time
    else:
        dp_exposed = dp_time  # always exposed in this model

    # PP bubble
    bubble = 0.0
    if cfg.pp > 1:
        bubble = pp_bubble_time(cfg.n_microbatches, cfg.pp, cfg.n_layers,
                                 fwd_layer + bwd_layer)

    # Total
    fwd_total = cfg.n_layers * fwd_layer
    bwd_total = cfg.n_layers * bwd_layer
    total = fwd_total + bwd_total + opt_time + dp_exposed + bubble

    # Overlapped term (for overlap=false comparison)
    overlapped_total = cfg.n_layers * overlapped_tp * 2  # fwd + bwd

    records = []
    # Forward layers
    for i in range(cfg.n_layers):
        records.append({"phase": "forward", "component": f"layer_{i}", "kind": "compute",
                        "seconds": fwd_layer, "predicted": True})
    # Backward layers
    for i in range(cfg.n_layers):
        records.append({"phase": "backward", "component": f"layer_{i}", "kind": "compute",
                        "seconds": bwd_layer, "predicted": True})
    # Optimizer
    records.append({"phase": "optimizer", "component": "optimizer", "kind": "compute",
                    "seconds": opt_time, "predicted": True})
    # DP allreduce
    records.append({"phase": "dp_sync", "component": "dp_allreduce", "kind": "comm",
                    "seconds": dp_exposed, "predicted": True})
    # Bubble
    if bubble > 0:
        records.append({"phase": "pp", "component": "bubble", "kind": "idle",
                        "seconds": bubble, "predicted": True})
    # Overlapped (hidden) term
    if overlapped_total > 0:
        records.append({"phase": "overlap", "component": "tp_allreduce_hidden", "kind": "hidden",
                        "seconds": overlapped_total, "predicted": True})

    return {
        "total_seconds": total,
        "overlapped_seconds": overlapped_total,
        "bubble_seconds": bubble,
        "dp_seconds": dp_exposed,
        "config": asdict(cfg),
        "overlap_policy": OVERLAP_POLICY,
        "overlap_provenance": OVERLAP_PROVENANCE,
        "components": records,
        "predicted": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="specs/h100_rack.json")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = StepConfig(
        n_layers=args.layers,
        n_microbatches=args.microbatches,
        dp=args.dp,
        tp=args.tp,
        pp=args.pp,
        overlap=not args.no_overlap,
    )
    result = predict_step(args.spec, cfg)

    lines = [json.dumps(r) for r in result["components"]]
    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")
    import sys
    print(f"total_seconds={result['total_seconds']:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
