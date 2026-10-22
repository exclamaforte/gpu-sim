"""cy0-L4: calibration harness -- refit parameters from trace JSONL."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

MIN_OBSERVATIONS = 3


@dataclass
class ParamHistory:
    name: str
    old_value: float
    new_value: float
    old_provenance: str
    new_provenance: str
    fit_method: str
    n_observations: int
    residuals_before: List[float]
    residuals_after: List[float]
    structural_finding: str = ""


def load_trace(path: str) -> List[Dict[str, Any]]:
    """Load trace JSONL records."""
    records = []
    for line in Path(path).read_text().strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def calibrate(trace_path: str, workload_id: str = "default") -> Dict[str, Any]:
    """Refit parameters from a trace. Returns param history."""
    records = load_trace(trace_path)
    if len(records) < MIN_OBSERVATIONS:
        raise ValueError(
            f"refusing to fit from {len(records)} observations "
            f"(minimum {MIN_OBSERVATIONS})"
        )

    # Group by op_class for compute efficiency refit
    by_class: Dict[str, List[Dict]] = {}
    for r in records:
        if r.get("kind") == "compute" and "op_class" in r:
            by_class.setdefault(r["op_class"], []).append(r)

    histories: List[ParamHistory] = []
    structural_findings: List[str] = []

    for cls, recs in by_class.items():
        # Estimate efficiency from residuals: if observed/predicted ratio is consistent,
        # refit efficiency = mean(predicted_seconds * eff / observed_seconds)
        old_eff = 0.5  # default
        # Compute residuals before
        residuals_before = []
        for r in recs:
            pred = r.get("seconds", 0)
            obs = r.get("observed_seconds", pred)  # if no observed, use predicted
            if pred > 0:
                residuals_before.append((obs - pred) / pred)

        # Simple refit: new_eff = old_eff * mean(predicted/observed) for compute
        ratios = []
        for r in recs:
            pred = r.get("seconds", 0)
            obs = r.get("observed_seconds", pred)
            if obs > 0 and pred > 0:
                ratios.append(pred / obs)
        new_eff = old_eff * (statistics.mean(ratios) if ratios else 1.0)
        new_eff = max(0.01, min(0.99, new_eff))

        residuals_after = []
        for r in recs:
            pred = r.get("seconds", 0)
            obs = r.get("observed_seconds", pred)
            if pred > 0:
                # Adjusted prediction
                adj_pred = pred * (old_eff / new_eff) if new_eff > 0 else pred
                residuals_after.append((obs - adj_pred) / adj_pred if adj_pred > 0 else 0.0)

        # Check if residuals can be explained by a single parameter
        if len(set(round(r, 2) for r in residuals_before)) > 3:
            structural_findings.append(
                f"op_class={cls}: residuals have >3 distinct values, "
                f"may require structural change"
            )

        histories.append(ParamHistory(
            name=f"eff_{cls}",
            old_value=old_eff,
            new_value=new_eff,
            old_provenance="published-spec assumption",
            new_provenance=f"fitted:{workload_id}",
            fit_method="mean ratio of predicted/observed",
            n_observations=len(recs),
            residuals_before=residuals_before,
            residuals_after=residuals_after,
        ))

    # Comm parameters: effective IB bandwidth
    comm_recs = [r for r in records if r.get("kind") == "comm"]
    if comm_recs:
        old_bw = 50.0  # NDR IB default GB/s
        ratios = []
        for r in comm_recs:
            pred = r.get("seconds", 0)
            obs = r.get("observed_seconds", pred)
            if obs > 0 and pred > 0:
                ratios.append(pred / obs)
        new_bw = old_bw * (statistics.mean(ratios) if ratios else 1.0)
        new_bw = max(1.0, new_bw)

        residuals_before = []
        residuals_after = []
        for r in comm_recs:
            pred = r.get("seconds", 0)
            obs = r.get("observed_seconds", pred)
            if pred > 0:
                residuals_before.append((obs - pred) / pred)
                adj = pred * (old_bw / new_bw) if new_bw > 0 else pred
                residuals_after.append((obs - adj) / adj if adj > 0 else 0.0)

        histories.append(ParamHistory(
            name="effective_ib_bw_gbs",
            old_value=old_bw,
            new_value=new_bw,
            old_provenance="published-spec assumption (NDR 400 Gbps)",
            new_provenance=f"fitted:{workload_id}",
            fit_method="mean ratio of predicted/observed",
            n_observations=len(comm_recs),
            residuals_before=residuals_before,
            residuals_after=residuals_after,
        ))

    return {
        "workload_id": workload_id,
        "n_observations": len(records),
        "param_histories": [asdict(h) for h in histories],
        "structural_findings": structural_findings,
        "predicted": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--workload-id", default="default")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = calibrate(args.trace, args.workload_id)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
