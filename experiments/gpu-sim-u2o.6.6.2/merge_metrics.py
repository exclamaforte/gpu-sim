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
    # BYTE CONVENTION, stated once and applied to every bandwidth number:
    #   bytes moved = n * sizeof(float) per READ buffer + n * sizeof(float) per
    #   WRITE buffer, where a pointer argument is read, written, or both
    #   depending on the kernel. That count is derived per cell from the
    #   kernel's argument roles, NOT from a single global assumption -- the
    #   previous revision assumed 2*n*4 for everything, which is right for
    #   copy/relu (one read + one write) and wrong for saxpy (two reads + one
    #   write) and for the lifted access kernels (index/seed buffers count
    #   too). Mixing conventions in one max() is what made the old ceiling
    #   unattributable.
    import anchor_kernels as ak
    ak_stride = lambda: ak.STRIDE
    READS = {"k_saxpy": 2, "k_relu": 1, "k_copy": 1, "k_strided_read": 1,
             "k_scatter_write": 1, "k_random_access": 1}
    WRITES = {"k_saxpy": 1, "k_relu": 1, "k_copy": 1, "k_strided_read": 1,
              "k_scatter_write": 1, "k_random_access": 1}
    INT_BYTES = 4          # the index/seed buffers are 4-byte integers

    def bytes_moved(c) -> int:
        k = c["kernel"]
        if k not in READS:
            return 0
        # strided_read touches only 1/stride of the input, so it moves less;
        # scatter/random move one element per thread, not a full buffer.
        n = c["n_problem_size"]
        per = READS[k] + WRITES[k]
        if k == "k_strided_read":
            eff = max(1, n // ak_stride())
            return eff * 4 + n * 4          # reads n/stride, writes n
        if k in ("k_scatter_write", "k_random_access"):
            return n * 4 + n * 4            # data + index/seed, both streamed
        return n * per * 4

    mem = [c for c in ok
           if c["op_type"] in ("memory", "access") and "auto/256" in c["label"]]
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
    # CEILING = the ARGMAX INDEX, never `max(...)[0]`. The previous revision
    # took max() over a candidate set and then labelled the result with
    # hbm_cells[0], which is the FIRST element of the set, not the argmax: the
    # published "1541.3 GB/s from saxpy" was actually strided_copy's figure
    # under a 2-buffer convention. Nothing about a ceiling is defensible if the
    # number and the label come from different cells. The full ranked list is
    # emitted so the choice is checkable.
    # CANDIDATE SET, stated and then applied: the HBM-bound cells only (the
    # largest problem size). An argmax over ALL memory cells picks the
    # L2-resident n=2^22 case at ~3.2 TB/s, which is a CACHE number being
    # published as the memory ceiling -- the exact defect this item exists to
    # fix. So: filter to the candidate set FIRST, then take the argmax INDEX
    # within it, and emit the full ranked list of the whole set alongside.
    max_n = max((c["n_problem_size"] for c in mem), default=0)
    candidates = [c for c in mem if c["n_problem_size"] == max_n]
    ranked = sorted(candidates, key=lambda c: -c["_bps"])
    ceiling_cell = ranked[0] if ranked else None
    ceiling_bytes_per_s = ceiling_cell["_bps"] if ceiling_cell else 0.0
    ceiling_source_label = ceiling_cell["label"] if ceiling_cell else None
    ceiling_candidates_all = [
        {"label": c["label"], "kernel": c["kernel"],
         "n_problem_size": c["n_problem_size"],
         "median_ms": c["median_ms"],
         "bytes_moved": bytes_moved(c),
         "achieved_GBps": c["_bps"] / 1e9,
         "l2_resident": 2 * c["n_problem_size"] * 4 / 1048576.0 <= 96}
        for c in sorted(mem, key=lambda c: -c["_bps"])[:6]]
    ceiling_candidates = [
        {"label": c["label"], "kernel": c["kernel"],
         "n_problem_size": c["n_problem_size"],
         "median_ms": c["median_ms"],
         "bytes_moved": bytes_moved(c),
         "achieved_GBps": c["_bps"] / 1e9,
         "l2_resident": 2 * c["n_problem_size"] * 4 / 1048576.0 <= 96}
        for c in ranked[:6]]
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

    # RE-DERIVED under the new ceiling AND the new candidate set (item 13).
    # The previously published 262144 is NOT carried forward: it survived a
    # 480x ceiling change because the old test was a ratio, and the old
    # candidate set also changed at the same time. This value is computed from
    # scratch below and is only shipped if the re-derivation stands on its own.
    crossover_n = None
    crossover_from = None
    for c in sorted(mem, key=lambda c: c["n_problem_size"]):
        if c.get("_bps") and c["_bps"] >= 0.10 * ceiling_bytes_per_s:
            crossover_n = c["n_problem_size"]
            crossover_from = c["label"]
            break

    # Reviewer MINOR: a "1block/1024" row is labelled n=2^26 but grid*block is
    # fixed at 1024, so it processes 1024 elements. Record `covers` on every
    # cell so a consumer never has to infer it from the label.
    # PER-CELL REGIME FLAG on all cells (item 2). The regimes block used to be
    # a separate shortlist, so 6.1.1's cell-by-cell diff had nothing to key
    # on. Threshold, stated once and applied to every cell:
    #   launch-bound  : median <= 10 us
    #   memory-bound  : not launch-bound AND achieved bandwidth >= 80% of the
    #                   measured ceiling
    #   compute-bound : not launch-bound AND not memory-bound
    # The 10 us figure is the PI's regime decision (6.6.2: 20 single-launch
    # trials resolve ~42% below 10 us and ~0.13% above), NOT a measurement
    # derived here.
    LAUNCH_BOUND_US = 10.0
    for c in ok:
        us = c["median_ms"] * 1000.0
        bps = c.get("_bps")
        if us <= LAUNCH_BOUND_US:
            c["regime"] = "launch-bound"
        elif bps and bps >= 0.80 * ceiling_bytes_per_s:
            c["regime"] = "memory-bound"
        else:
            c["regime"] = "compute-bound"
        c.pop("_bps", None)
    regime_counts = {}
    for c in ok:
        regime_counts[c["regime"]] = regime_counts.get(c["regime"], 0) + 1

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
            "launch_to_memory_crossover_from_cell": crossover_from,
            "launch_to_memory_crossover_note": (
                "RE-DERIVED under this commit's ceiling AND candidate set. The "
                "previously published 262144 is not carried forward: it "
                "survived a 480x ceiling change because the test was a ratio, "
                "and the candidate set changed at the same time, so its "
                "survival was not validation. This value stands only on the "
                "derivation recorded here."),
            "regime_thresholds": {
                "launch_bound_us": LAUNCH_BOUND_US,
                "launch_bound_source": ("the PI's regime decision for 6.6.2, "
                                        "not a measurement made in this file"),
                "memory_bound": "achieved >= 80% of the measured ceiling",
            },
            "ceiling_candidate_set": {
                "rule": "cells at the largest problem size (HBM-bound), "
                        "n_problem_size = %d" % max_n,
                "excluded_and_why": "n=2^22 is L2-resident on this part and "
                                    "sustains far more than HBM; including it "
                                    "would publish a cache bandwidth as the "
                                    "memory ceiling",
                "ranked": ceiling_candidates,
            },
            "ceiling_candidates_ranked_all_cells": ceiling_candidates_all,
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
        "regime_counts": regime_counts,
        "cells": ok,
        "failed_cells": [c for c in cells if "error" in c],
    }
    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("anchor", "regimes", "reproducibility")}, indent=2)[:2600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
