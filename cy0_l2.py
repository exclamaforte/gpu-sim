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


def layer_compute_time(spec_path: str) -> float:
    """Compute time for one layer (sum of compute ops, no comm)."""
    result = cy0_l1.predict_layer(spec_path, tp=1)  # tp=1: no TP comm in layer
    return sum(c["seconds"] for c in result["components"] if c["kind"] == "compute")


def layer_tp_comm_time(spec_path: str, tp: int) -> float:
    """TP allreduce time for one layer."""
    result = cy0_l1.predict_layer(spec_path, tp=tp)
    comm = [c for c in result["components"] if c["kind"] == "comm"]
    return comm[0]["seconds"] if comm else 0.0


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


def pp_bubble_time(n_microbatches: int, n_stages: int, layer_time: float) -> float:
    """Pipeline bubble: (n_stages - 1) * layer_time for fill + (n_stages - 1) * layer_time for drain.
    With m microbatches, bubble = (n_stages - 1) * (layer_time / m) * 2? 
    Standard: bubble = (pp - 1) / (m + pp - 1) * total_time.
    We return the absolute bubble time: (pp-1) * layer_time for a single layer pass.
    """
    if n_stages <= 1:
        return 0.0
    # Simple model: fill + drain = (pp-1) microbatches of idle time
    return (n_stages - 1) * layer_time


def predict_step(spec_path: str, cfg: StepConfig) -> Dict[str, Any]:
    """Predict one training step."""
    spec = load_spec(spec_path)
    lc = layer_compute_time(spec_path)
    tp_comm = layer_tp_comm_time(spec_path, cfg.tp)

    # Per-layer forward time
    if cfg.overlap and OVERLAP_POLICY["tp_allreduce_hidden"]:
        fwd_layer = lc  # TP comm hidden
        overlapped_tp = tp_comm
    else:
        fwd_layer = lc + tp_comm
        overlapped_tp = 0.0

    # Backward: same compute + TP comm (gradient)
    if cfg.overlap and OVERLAP_POLICY["tp_allreduce_hidden"]:
        bwd_layer = lc
    else:
        bwd_layer = lc + tp_comm

    # Optimizer: simple, ~10% of one layer's compute
    opt_time = lc * 0.1

    # DP allreduce: after backward, gradient sync
    # Gradient size ~ hidden * hidden * 4 bytes (simplified)
    grad_bytes = cy0_l1.HIDDEN * cy0_l1.HIDDEN * 4
    dp_time = dp_allreduce_time(spec_path, cfg.dp, grad_bytes)
    if cfg.overlap and OVERLAP_POLICY["dp_allreduce_exposed"]:
        dp_exposed = dp_time
    else:
        dp_exposed = dp_time  # always exposed in this model

    # PP bubble
    bubble = 0.0
    if cfg.pp > 1:
        bubble = pp_bubble_time(cfg.n_microbatches, cfg.pp, fwd_layer + bwd_layer)

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
