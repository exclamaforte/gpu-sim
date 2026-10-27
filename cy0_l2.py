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
    ep: int = 1   # expert parallelism degree (trace world size)
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
    return dp_allreduce_detail(spec_path, dp, grad_bytes).get("duration_s", 0.0)


def tp_allreduce_detail(spec_path: str, tp: int, cfg: "StepConfig" = None) -> Dict[str, Any]:
    """Full link_model result for the TP allreduce (one collective)."""
    if tp <= 1:
        return {"duration_s": 0.0, "algorithm": "ring_allreduce",
                "bottleneck": {"link_id": None, "kind": None},
                "regime": "bandwidth", "bytes": 0}
    kw = {}
    if cfg is not None:
        kw = {"hidden": cfg.hidden, "ffn": cfg.ffn, "seq": cfg.seq, "batch": cfg.batch}
    result = cy0_l1.predict_layer(spec_path, tp=tp, **kw)
    comm = [c for c in result["components"] if c["kind"] == "comm"]
    ar_bytes = 0
    # derive bytes from cy0_l1 layer ops (tp_allreduce)
    try:
        ops = cy0_l1.layer_ops(hidden=cfg.hidden if cfg else cy0_l1.HIDDEN,
                               ffn=cfg.ffn if cfg else cy0_l1.FFN_DIM,
                               seq=cfg.seq if cfg else cy0_l1.SEQ_LEN,
                               batch=cfg.batch if cfg else cy0_l1.BATCH)
        ar_bytes = [o.bytes for o in ops if o.name == "tp_allreduce"][0]
    except Exception:
        ar_bytes = 0
    count = ar_bytes // 2 if ar_bytes else 0
    # call link_model directly for bottleneck/regime
    try:
        spec = load_spec(spec_path)
        res = predict_collective(spec, "ring_allreduce", count, "bfloat16",
                                n_ranks=tp, pieces_per_phase=8)
        res = dict(res)
        res["bytes"] = ar_bytes
        res["count"] = count
        res["dtype"] = "bfloat16"
        return res
    except Exception:
        return {"duration_s": comm[0]["seconds"] if comm else 0.0,
                "algorithm": "ring_allreduce",
                "bottleneck": {"link_id": None, "kind": None},
                "regime": "bandwidth", "bytes": ar_bytes, "count": count,
                "dtype": "bfloat16"}


def dp_allreduce_detail(spec_path: str, dp: int, grad_bytes: int) -> Dict[str, Any]:
    """Full link_model result for the DP allreduce."""
    if dp <= 1:
        return {"duration_s": 0.0, "algorithm": "ring_allreduce",
                "bottleneck": {"link_id": None, "kind": None},
                "regime": "bandwidth", "bytes": grad_bytes, "count": grad_bytes // 4,
                "dtype": "float32"}
    count = grad_bytes // 4
    spec = load_spec(spec_path)
    res = predict_collective(spec, "ring_allreduce", count, "float32",
                            n_ranks=dp, pieces_per_phase=8)
    res = dict(res)
    res["bytes"] = grad_bytes
    res["count"] = count
    res["dtype"] = "float32"
    return res


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
    tp_detail = tp_allreduce_detail(spec_path, cfg.tp, cfg)

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

    # Optimizer: Adam, ~12 FLOPs per parameter per step
    # params per layer = 4*H^2 + 3*H*F (QKV, output, gate, up, down, norms)
    H = cy0_l1.HIDDEN
    F = cy0_l1.FFN_DIM
    params_per_layer = 4 * H * H + 3 * H * F
    opt_flops = 12 * params_per_layer * cfg.n_layers
    # peak_flops from spec (FLOPS)
    spec = load_spec(spec_path)
    pf = cy0_l1.peak_flops(spec) * 1e12
    eff = 0.30  # elementwise efficiency (assumption, refit in L4)
    opt_time = opt_flops / (pf * eff)

    # DP allreduce: after backward, gradient sync
    # Gradient size = dtype_bytes (fp32=4) * params_per_layer * n_layers
    H = cy0_l1.HIDDEN
    F = cy0_l1.FFN_DIM
    params_per_layer = 4 * H * H + 3 * H * F
    grad_bytes = 4 * params_per_layer * cfg.n_layers  # fp32 gradients
    dp_detail = dp_allreduce_detail(spec_path, cfg.dp, grad_bytes)
    dp_time = dp_detail.get("duration_s", 0.0)
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

    # --- cy0-L6: rank-resolved trace timeline (additive, does not change components) ---
    ws = cfg.dp * cfg.tp * cfg.pp * getattr(cfg, "ep", 1)
    tp_bl = (tp_detail.get("bottleneck") or {}).get("link_id")
    tp_regime = tp_detail.get("regime", "bandwidth")
    tp_bytes = int(tp_detail.get("bytes") or 0)
    tp_count = int(tp_detail.get("count") or (tp_bytes // 2))
    dp_bl = (dp_detail.get("bottleneck") or {}).get("link_id")
    dp_regime = dp_detail.get("regime", "bandwidth")
    dp_bytes_i = int(dp_detail.get("bytes") or grad_bytes)
    dp_count_i = int(dp_detail.get("count") or (grad_bytes // 4))
    dp_algo = "ring_allreduce"

    timeline = []
    timeline.append({
        "type": "header",
        "world_size": ws,
        "placement": "TP group on one node, ranks contiguous per cy0_l3 (ranks 0..world_size-1).",
        "reconciliation": "sum of record seconds per rank excluding kind=hidden equals total_seconds; overlapped TP comm is reported hidden and not double-counted.",
        "provenance_policy": "published-spec | measured:<artifact> | fitted:<trace>",
        "predicted": True,
    })

    def _ns(x):
        return int(round(float(x) * 1e9))

    for r in range(ws):
        ts = 0
        # forward layers in order
        for i in range(cfg.n_layers):
            dur = fwd_layer
            timeline.append({"ts_ns": ts, "rank": r, "nranks": ws,
                             "op": f"forward_layer_{i}", "kind": "compute",
                             "phase": "forward", "seconds": dur, "predicted_s": dur,
                             "predicted": True,
                             "provenance": "published-spec: cy0_l2 fwd_layer"})
            ts += _ns(dur)
        # backward layers in reverse
        for i in range(cfg.n_layers - 1, -1, -1):
            dur = bwd_layer
            timeline.append({"ts_ns": ts, "rank": r, "nranks": ws,
                             "op": f"backward_layer_{i}", "kind": "compute",
                             "phase": "backward", "seconds": dur, "predicted_s": dur,
                             "predicted": True,
                             "provenance": "published-spec: cy0_l2 bwd_layer"})
            ts += _ns(dur)
        # optimizer
        timeline.append({"ts_ns": ts, "rank": r, "nranks": ws,
                         "op": "optimizer", "kind": "compute", "phase": "optimizer",
                         "seconds": opt_time, "predicted_s": opt_time,
                         "predicted": True,
                         "provenance": "published-spec: cy0_l2 optimizer (12 FLOP/param)"})
        ts += _ns(opt_time)
        # dp sync collective
        timeline.append({"ts_ns": ts, "rank": r, "nranks": ws,
                         "op": "allreduce", "kind": "comm", "phase": "dp_sync",
                         "count": dp_count_i, "dtype": "float32", "bytes": dp_bytes_i,
                         "algorithm": dp_algo, "predicted_s": dp_exposed,
                         "seconds": dp_exposed, "bottleneck_link": dp_bl,
                         "regime": dp_regime, "predicted": True,
                         "provenance": "published-spec: link_model.predict ring_allreduce"})
        ts += _ns(dp_exposed)
        # bubble (idle)
        if bubble > 0:
            timeline.append({"ts_ns": ts, "rank": r, "nranks": ws,
                             "op": "bubble", "kind": "idle", "phase": "pp",
                             "seconds": bubble, "predicted_s": bubble,
                             "predicted": True,
                             "provenance": "published-spec: cy0_l2 pp_bubble"})
            ts += _ns(bubble)
        # hidden TP allreduce (overlapped, not on critical path -> no ts advance)
        if overlapped_total > 0:
            timeline.append({"ts_ns": ts, "rank": r, "nranks": ws,
                             "op": "allreduce", "kind": "hidden", "phase": "overlap",
                             "count": tp_count, "dtype": "bfloat16", "bytes": tp_bytes,
                             "algorithm": "ring_allreduce", "predicted_s": overlapped_total,
                             "seconds": overlapped_total, "bottleneck_link": tp_bl,
                             "regime": tp_regime, "predicted": True,
                             "provenance": "published-spec: link_model.predict ring_allreduce (hidden)"})

    return {
        "total_seconds": total,
        "overlapped_seconds": overlapped_total,
        "bubble_seconds": bubble,
        "dp_seconds": dp_exposed,
        "config": asdict(cfg),
        "overlap_policy": OVERLAP_POLICY,
        "overlap_provenance": OVERLAP_PROVENANCE,
        "components": records,
        "timeline": timeline,
        "trace": timeline,
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
