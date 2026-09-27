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
    # memory-bound: at or near the HBM ceiling.
    # compute-bound: achieved bytes/s is a large fraction of the ceiling.
    # The CEILING is taken from the HBM-bound cells (the largest n), NOT from
    # the best cell. The reviewer was right: the previous revision took the
    # ceiling from the n=2^22 cell, which is L2-resident on this part and so
    # reported a cache bandwidth (~3.2 TB/s) as the memory ceiling -- every
    # later roofline would then be calibrated against cache. The HBM-bound
    # n=2^26 cell sustains ~1.5 TB/s, which is the number a roofline wants.
    # L2 residency is recorded per cell so a consumer can see which is which.
    # measured elementwise case achieves (the memory-bound ceiling here).
    mem = [c for c in ok if c["op_type"] == "memory" and "auto/256" in c["label"]]
    ceiling_bytes_per_s = 0.0
    for c in mem:
        n = c["n_problem_size"]
        moved = 2 * n * 4          # one read + one write, fp32
        # median_ms is MILLIseconds, so bytes/ms is 1000x short of bytes/s.
        # The previous revision divided by 1e9 on top of that and reported
        # bandwidths 1000x too small (3.2 "GB/s" for a case that actually
        # sustains ~3.2 TB/s). The reviewer's unit finding was correct.
        bps = moved / (c["median_ms"] / 1000.0)
        # remember the per-cell figure; the ceiling is picked below from the
        # largest (HBM-bound) n, not from the maximum
        c["_bps"] = bps
        c["_working_set_mb"] = 2 * n * 4 / 1048576.0
    hbm_cells = [c for c in mem if c["n_problem_size"] == max(x["n_problem_size"] for x in mem)]
    ceiling_bytes_per_s = max(c["_bps"] for c in hbm_cells) if hbm_cells else 0.0
    ceiling_source_label = hbm_cells[0]["label"] if hbm_cells else None
    launch_bound, memory_bound = [], []
    for c in mem:
        us = c["median_ms"] * 1000.0
        n = c["n_problem_size"]
        bps = c["_bps"]
        entry = {"label": c["label"], "n_problem_size": n,
                 # working set, so a consumer can tell an L2-resident cell
                 # (n=2^22: 2*4 MiB = 32 MiB of traffic) from an HBM-bound
                 # one (n=2^26: 512 MiB) without recomputing anything
                 "working_set_MB": c["_working_set_mb"],
                 "l2_resident": c["_working_set_mb"] <= 96,
                 "median_us": us, "achieved_GBps": bps / 1e9}
        if bps < 0.10 * ceiling_bytes_per_s:
            launch_bound.append(entry)
        elif bps >= 0.80 * ceiling_bytes_per_s:
            memory_bound.append(entry)

    crossover_n = None
    for c in sorted(mem, key=lambda c: c["n_problem_size"]):
        if c["_bps"] >= 0.10 * ceiling_bytes_per_s:
            crossover_n = c["n_problem_size"]
            break

    # Reviewer MINOR: a "1block/1024" row is labelled n=2^26 but grid*block is
    # fixed at 1024, so it processes 1024 elements. Record `covers` on every
    # cell so a consumer never has to infer it from the label.
    for c in ok:
        g = c.get("grid") or [1, 1, 1]
        b = c.get("block") or [1, 1, 1]
        threads = (g[0] * g[1] * g[2]) * (b[0] * b[1] * b[2])
        c["covers"] = min(threads, c["n_problem_size"])
        c["covers_note"] = (
            "elements actually processed = min(n_problem_size, grid*block); "
            "the deliberately-poor 1block/1024 configs cover 1024 elements "
            "whatever the label says")

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
            "memory_ceiling_source_cell": ceiling_source_label,
            "memory_ceiling_definition": ("best achieved bytes/s among the "
                                          "HBM-bound cells (the largest n, "
                                          "2^26), 2*n*4 bytes moved (one read + "
                                          "one write, fp32). NOT the best cell "
                                          "overall: the n=2^22 cell is "
                                          "L2-resident on this part and sustains "
                                          "far more, so taking the maximum would "
                                          "publish a cache bandwidth as the "
                                          "memory ceiling and calibrate every "
                                          "later roofline against cache."),
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
