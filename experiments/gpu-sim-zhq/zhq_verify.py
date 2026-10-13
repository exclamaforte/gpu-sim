#!/usr/bin/env python3
"""gpu-sim-zhq (CPU-only): latency-arm fix before/after on 456 rows.

Unlike g8x's gate, this one FAILS LOUDLY when predictions do not change
(acceptance 1): a byte-identical before/after is the defect's signature.

Loads the PRE-FIX roofline from git (HEAD:epic_u2o/roofline.py) and the fixed
one from the tree, rebuilds each row's features deterministically
(regenerate source -> extract_features), predicts both, and reports per-op
error before/after. No numpy (not installed here); stdlib statistics only.
Usage: python3 experiments/gpu-sim-g8x/baseline_rerun.py --metrics-out <p>
"""
import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "epic_u2o"))

from epic_u2o.generator_stage1 import generate_kernels  # noqa: E402
from epic_u2o.features import LaunchConfig, extract_features  # noqa: E402
import epic_u2o.roofline as roof_new  # noqa: E402


# Reviewer pin (g8x rework): the pre-fix reference is a FIXED merged commit,
# not HEAD. HEAD moves (rebase, merge); a gate that loads 'before' from a
# moving ref can silently compare the fix against itself and pass on
# byte-identical metrics. 46ddfa9 is the merged master the g8x branch was
# cut from, whose roofline.py still ships the three inflating constants.
# zhq compares against the g8x-merged master (occupancy fixed,
# latency arm still invented): the change measured here is the latency
# arm alone. Full hash below; merged and immutable.
PRE_FIX_REF = "15c4c7331b1823a48445d92935f46572a96a0881"


def load_old_roofline():
    src = subprocess.run(["git", "show", f"{PRE_FIX_REF}:epic_u2o/roofline.py"],
                         cwd=str(REPO), capture_output=True, text=True,
                         check=True).stdout
    # Both directions: the old markers present AND the fix absent. Either
    # check alone passes on the wrong tree (a fixed tree keeps no old
    # markers; an unrelated old tree may lack the fix for other reasons --
    # the pair pins THIS ref, not just 'some old file').
    assert "sync_overhead_us" in src, "pre-zhq marker gone: wrong ref?"
    assert "launch_overhead_us = 10.0" in src, "pre-zhq marker gone: wrong ref?"
    assert "LAUNCH_FLOOR_US" not in src, "zhq fix already in pre-fix ref?!"
    spec = importlib.util.spec_from_loader("roofline_before", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "roofline_before.py", "exec"), mod.__dict__)
    return mod


def row_features(row):
    g, b = row["launch_config"]["grid"], row["launch_config"]["block"]
    ss = generate_kernels(ops=[row["op_type"]], grid_sweeps=[g],
                          block_sweeps=[b], problem_sizes=[row["problem_size"]])
    src = [s for s in ss if s.kernel_name == row["kernel_name"]][0].source
    lc = LaunchConfig(grid_x=g[0], grid_y=g[1], grid_z=g[2], block_x=b[0],
                      block_y=b[1], block_z=b[2],
                      shared_mem_bytes=row["launch_config"]["shared_mem_bytes"])
    return extract_features(src, lc)


def ac1_gate(mean_move, bn):
    """AC1 as a pure function, so the suite can prove BOTH halves fire.

    Returns the failures list. A gate half demonstrated only against passing
    data is not a gate half (this function's movement branch shipped dead
    behind a rebound `fails = []`, caught by review, not by any test)."""
    fails = []
    if mean_move <= 0.01:
        fails.append(f"AC1: mean prediction movement {mean_move:.4f} <= 1% -- "
                     "the fix changed nothing; failing loudly")
    if set(bn) == {"latency->latency"}:
        fails.append("AC1: bottleneck map still latency->latency everywhere -- "
                     "the fix changed no decision; failing loudly")
    return fails


def to_new(f):
    d = f.to_dict() if hasattr(f, "to_dict") else dict(f.__dict__)
    return roof_new.KernelFeatures(**{k: d[k] for k in
                                      roof_new.KernelFeatures.__dataclass_fields__})


def to_old(mod, f):
    d = f.to_dict() if hasattr(f, "to_dict") else dict(f.__dict__)
    return mod.KernelFeatures(**{k: d[k] for k in
                                 mod.KernelFeatures.__dataclass_fields__})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    old = load_old_roofline()
    rows = []
    for c in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl"):
        rows += [json.loads(l) for l in (REPO / c).read_text().splitlines()]
    per_op, tot_b, tot_a, bias_b, bias_a = defaultdict(lambda: [[], []]), [], [], [], []
    occ_b, occ_a, bn, moves = [], [], {}, []
    for i, r in enumerate(rows):
        f = row_features(r)
        rb = old.roofline_predict(to_old(old, f))
        ra = roof_new.roofline_predict(to_new(f))
        occ_b.append(rb.achieved_occupancy)
        occ_a.append(ra.achieved_occupancy)
        bn[f"{rb.bottleneck}->{ra.bottleneck}"] = bn.get(f"{rb.bottleneck}->{ra.bottleneck}", 0) + 1
        pb, pa = rb.predicted_time_ms, ra.predicted_time_ms
        moves.append(abs(pa - pb) / pb if pb else 0.0)
        actual = r["median_runtime_ms"]
        if actual <= 0:
            continue
        eb, ea = (pb - actual) / actual, (pa - actual) / actual
        per_op[r["op_type"]][0].append(abs(eb))
        per_op[r["op_type"]][1].append(abs(ea))
        tot_b.append(abs(eb))
        tot_a.append(abs(ea))
        bias_b.append(eb)
        bias_a.append(ea)
    ops = {op: {"n": len(v[0]),
                "mean_abs_before": round(statistics.fmean(v[0]), 4),
                "mean_abs_after": round(statistics.fmean(v[1]), 4)}
           for op, v in sorted(per_op.items())}
    # ACCEPTANCE 1: the gate FAILS when predictions do not change. g8x
    # reported byte-identical 7.7238 -> 7.7238 as a pass; that exact shape
    # -- tiny prediction movement with a latency-only bottleneck map -- is
    # now an explicit failure, not a verdict. Thresholds: mean abs
    # prediction movement must exceed 1% relative, and the bottleneck map
    # must stop being latency->latency everywhere.
    mean_move = statistics.fmean(moves)
    fails = ac1_gate(mean_move, bn)
    metrics = {
        "bead": "gpu-sim-zhq", "n_rows": len(rows),
        "overall_mean_abs_before": round(statistics.fmean(tot_b), 4),
        "overall_mean_abs_after": round(statistics.fmean(tot_a), 4),
        "signed_bias_before": round(statistics.fmean(bias_b), 4),
        "signed_bias_after": round(statistics.fmean(bias_a), 4),
        "per_op": ops,
        "mean_occupancy_before": round(statistics.fmean(occ_b), 4),
        "mean_occupancy_after": round(statistics.fmean(occ_a), 4),
        "bottleneck_before_after": bn,
        "launch_floor_us": roof_new.hw_5090.LAUNCH_FLOOR_US,
        "launch_source": ("6.6.1 metrics.json noise_floor, max of sub-2^18 "
                            "floors; ts final job 20260927t162535-cb00ea78"),
        "deleted_terms": ["sync_overhead_us = grid*0.01", "wave_overhead_us = (waves-1)*5.0"],
        "mean_prediction_movement": round(mean_move, 4),
        "verdict": "pass" if not fails else "fail", "failures": fails}
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"n={len(rows)} abs {metrics['overall_mean_abs_before']} -> "
          f"{metrics['overall_mean_abs_after']} bias {metrics['signed_bias_before']} -> "
          f"{metrics['signed_bias_after']} bottlenecks={bn} verdict={metrics['verdict']} -> {args.metrics_out}",
          flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
