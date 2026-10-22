"""cy0-L3: parallelism planner + predict_plan CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cy0_l1
import cy0_l2
from cy0_l2 import StepConfig, predict_step
from epic_cluster.spec import load as load_spec


class PlanError(ValueError):
    """Refusal: incoherent plan."""


def plan(model_cfg: Dict[str, Any], spec_path: str, dp: int, tp: int, pp: int,
         ep: int = 1, zero_stage: int = 0, seq: int = 8192, batch: int = 16) -> Dict[str, Any]:
    """Validate and build an annotated plan."""
    spec = load_spec(spec_path)
    nodes = spec["nodes"]
    n_nodes = len(nodes)
    gpus_per_node = len(nodes[0]["devices"])
    total_gpus = n_nodes * gpus_per_node
    world_size = dp * tp * pp * ep

    # Refusal: world size must not exceed total GPUs
    if world_size > total_gpus:
        raise PlanError(
            f"world_size={world_size} (dp={dp}*tp={tp}*pp={pp}*ep={ep}) "
            f"> total_gpus={total_gpus} ({n_nodes} nodes x {gpus_per_node} GPUs)"
        )

    # Refusal: tp must divide total GPUs
    if total_gpus % tp != 0:
        raise PlanError(
            f"tp={tp} does not divide total_gpus={total_gpus}"
        )

    # Placement check: do all ranks in a TP group reside on the same node?
    # A TP group of size tp uses ranks 0..tp-1 (contiguous placement assumed).
    # If any of those ranks is on a different node, TP spans nodes.
    tp_spans_nodes = False
    tp_nodes = set()
    for r in range(tp):
        if r < len(spec["ranks"]):
            tp_nodes.add(spec["ranks"][r]["node"])
    if len(tp_nodes) > 1:
        tp_spans_nodes = True

    warnings = []
    if tp_spans_nodes:
        warnings.append(
            f"WARNING: tp={tp} spans multiple nodes {sorted(tp_nodes)}, "
            f"forcing IB into the latency-critical path"
        )

    # Memory check: model params per device
    model_bytes = model_cfg.get("param_bytes", 7e9 * 2)  # default 7B params in bf16
    # Simplified: params / (tp * dp) must fit in mem_bytes
    per_device_params = model_bytes / (tp * (1 if zero_stage == 0 else dp))
    model_mem = list(spec["models"].values())[0]
    if per_device_params > model_mem["mem_bytes"] * 0.9:
        raise PlanError(
            f"model does not fit: {per_device_params/1e9:.1f} GB params per device "
            f"> 90% of {model_mem['mem_bytes']/1e9:.1f} GB device memory"
        )

    cfg = StepConfig(n_layers=model_cfg.get("n_layers", 2),
                     n_microbatches=batch,
                     dp=dp, tp=tp, pp=pp,
                     overlap=True)
    # Override seq/batch in layer model by monkey-patching cy0_l1 globals
    import cy0_l1
    cy0_l1.SEQ_LEN = seq
    cy0_l1.BATCH = 1  # per-microbatch batch
    result = predict_step(spec_path, cfg)

    return {
        "total_seconds": result["total_seconds"],
        "bubble_seconds": result["bubble_seconds"],
        "dp_seconds": result["dp_seconds"],
        "overlapped_seconds": result["overlapped_seconds"],
        "components": result["components"],
        "config": {
            "dp": dp, "tp": tp, "pp": pp, "ep": ep,
            "zero_stage": zero_stage, "seq": seq, "batch": batch,
        },
        "warnings": warnings,
        "predicted": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="specs/h100_rack.json")
    parser.add_argument("--model", default="default")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--zero-stage", type=int, default=0, dest="zero_stage")
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    model_cfg = {"n_layers": 2, "param_bytes": 7e9 * 2}
    try:
        result = plan(model_cfg, args.spec, args.dp, args.tp, args.pp,
                      args.ep, args.zero_stage, args.seq, args.batch)
    except PlanError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(f"total_seconds={result['total_seconds']:.6f}")
    print(f"  compute: {sum(c['seconds'] for c in result['components'] if c['kind']=='compute'):.6f}")
    print(f"  TP comm (hidden): {result['overlapped_seconds']:.6f}")
    print(f"  DP comm (exposed): {result['dp_seconds']:.6f}")
    print(f"  PP bubble: {result['bubble_seconds']:.6f}")
    for w in result["warnings"]:
        print(f"  {w}")

    if args.json:
        Path(args.json).write_text("\n".join(json.dumps(c) for c in result["components"]) + "\n")
        print(f"trace written to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
