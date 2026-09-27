#!/usr/bin/env python3
"""Merge the per-family anchor files into the canonical 6.6.2 metrics.json.

CPU-only post-processing: no GPU, no re-measurement. Produces the top-level
`anchor` and `regimes` blocks the bead's acceptance asks for, both derived
from the measured cells rather than from theory.
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILIES = ("launch", "memory", "reduce", "gemm")


def main() -> int:
    fams, cells, tel = {}, [], []
    for fam in FAMILIES:
        p = HERE / f"family_{fam}.json"
        if not p.exists():
            print(f"MISSING {p}")
            return 1
        d = json.loads(p.read_text())
        fams[fam] = d
        cells.extend(d["results"])
        tel.append((d["telemetry_before"], d["telemetry_after"]))

    ok = [c for c in cells if "error" not in c]

    # ---- anchor block ---------------------------------------------------
    empties = [c for c in ok if c["kernel"] == "k_empty"]
    launch_us = statistics.median([c["median_ms"] for c in empties]) * 1000.0

    best = {}
    for c in ok:
        key = c["op_type"]
        if key not in best or c["median_ms"] < best[key]["median_ms"]:
            best[key] = c

    # ---- regimes, from the data ----------------------------------------
    # launch-bound: the apparent bandwidth is far below what the hardware
    # actually sustains, i.e. the time is not moving the data at all. Using
    # "within 10% of the empty-kernel median" does NOT work here: single-launch
    # timing adds ~1 us of measurement overhead on top of the ~1.7 us launch
    # cost (gpu-sim-u2o.6.6.1), so every small-n cell lands near 2.7 us and
    # would be mislabelled. The bandwidth-ratio test is the honest one.
    # memory-bound: at or near the measured ceiling.
    # compute-bound: achieved bytes/s is a large fraction of what the largest
    # measured elementwise case achieves (the memory-bound ceiling here).
    mem = [c for c in ok if c["op_type"] == "memory" and "auto/256" in c["label"]]
    ceiling_bytes_per_s = 0.0
    for c in mem:
        n = c["n_problem_size"]
        moved = 2 * n * 4          # one read + one write, fp32
        bps = moved / c["median_ms"]
        ceiling_bytes_per_s = max(ceiling_bytes_per_s, bps)
    launch_bound, memory_bound = [], []
    for c in mem:
        us = c["median_ms"] * 1000.0
        n = c["n_problem_size"]
        bps = (2 * n * 4) / c["median_ms"]
        entry = {"label": c["label"], "n_problem_size": n,
                 "median_us": us, "achieved_GBps": bps / 1e9}
        if bps < 0.10 * ceiling_bytes_per_s:
            launch_bound.append(entry)
        elif bps >= 0.80 * ceiling_bytes_per_s:
            memory_bound.append(entry)

    crossover_n = None
    for c in sorted(mem, key=lambda c: c["n_problem_size"]):
        if (2 * c["n_problem_size"] * 4) / c["median_ms"] >= 0.10 * ceiling_bytes_per_s:
            crossover_n = c["n_problem_size"]
            break

    ai_cvs = [c["across_invocation_cv"] for c in ok
              if c.get("across_invocation_cv") is not None]
    small = [c["across_invocation_cv"] for c in ok
             if c.get("across_invocation_cv") is not None
             and c["median_ms"] < 0.01]
    large = [c["across_invocation_cv"] for c in ok
             if c.get("across_invocation_cv") is not None
             and c["median_ms"] >= 0.01]

    out = {
        "bead": "gpu-sim-u2o.6.6.2",
        "what": ("anchor + scaling set measured on the CURRENT measurement path "
                 "(synthesize -> compile_kernel -> run_kernel) BEFORE the "
                 "6.1.3/6.1.1 rework, so 6.1.1's closing run can be diffed "
                 "against it"),
        "preflight_bypassed": True,
        "preflight_bypass_reason": ("preflight_check executes on the real GPU "
                                    "(PI ruling 6.1.1); an anchor must not carry "
                                    "an unaccounted extra execution"),
        "kernels": "hand-written in anchor_kernels.py, NOT generator output "
                   "(gpu-sim-9ti: stage2-5 kernel_name is untrustworthy)",
        "design": {"trials_per_invocation": 100, "invocations": 3,
                   "invocations_are": "independent PROCESSES",
                   "n_configs": len(ok)},
        "telemetry": [{"family": f, "before": b, "after": a}
                      for f, (b, a) in zip(FAMILIES, tel)],
        "throttling_observed": any(f["throttling_observed"] for f in fams.values()),
        "anchor": {
            "empty_kernel_launch_cost_us": launch_us,
            "empty_kernel_note": ("median over every k_empty config and all 3 "
                                  "invocations; single-launch cudaEvent timing, "
                                  "so it includes the measurement overhead that "
                                  "gpu-sim-u2o.6.6.1 decomposed (~0.4-1.4 us)"),
            "best_known_config_per_family": {
                k: {"label": v["label"], "kernel": v["kernel"],
                    "n_problem_size": v["n_problem_size"],
                    "grid": v["grid"], "block": v["block"],
                    "median_ms": v["median_ms"],
                    "across_invocation_cv": v["across_invocation_cv"]}
                for k, v in best.items()},
        },
        "regimes": {
            "launch_bound_cells": launch_bound,
            "memory_bound_cells": memory_bound,
            "launch_to_memory_crossover_n": crossover_n,
            "crossover_definition": ("smallest measured n (auto/256 elementwise) whose "
                                     "achieved bandwidth reaches 10% of the measured "
                                     "ceiling; below it the measurement is launch "
                                     "overhead, not data movement -- from the data, "
                                     "not theory"),
            "measured_memory_ceiling_GBps": ceiling_bytes_per_s / 1e9,
            "memory_ceiling_definition": ("best achieved bytes/s over the "
                                          "auto/256 elementwise cells, 2*n*4 bytes "
                                          "moved (one read + one write, fp32)"),
        },
        "reproducibility": {
            "across_invocation_cv_median_all": statistics.median(ai_cvs) if ai_cvs else None,
            "across_invocation_cv_median_sub10us": statistics.median(small) if small else None,
            "across_invocation_cv_median_over10us": statistics.median(large) if large else None,
            "note": ("the headline number for this bead: within-process cv badly "
                     "understates the spread of the reported median. Compare with "
                     "the `cv` field, which is within-process only."),
        },
        "cells": ok,
        "failed_cells": [c for c in cells if "error" in c],
    }
    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("anchor", "regimes", "reproducibility")}, indent=2)[:2600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
