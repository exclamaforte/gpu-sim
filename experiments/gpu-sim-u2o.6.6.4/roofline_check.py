#!/usr/bin/env python3
"""gpu-sim-u2o.6.6.4 -- does the analytic roofline predict our measurements?

Pure CPU, no GPU, no ts job, no harness changes. The point is to TEST a
decision, not to measure anything: the PI's regime rule ("weight data
generation at >= 10 us") currently rests on two numbers, and this asks whether
an analytic model -- the thing 9.4/9.5 will be compared against -- is usable
below that line or not.

Two forms, per cell:
  t_roofline          = min(flops / peak_flops, bytes / bandwidth_ceiling)
  t_launch_corrected  = t_launch(grid, block) + t_roofline

with t_launch taken from the anchor's OWN k_empty / k_noop_store cells at
matching launch configurations, fitted if the grid/block is not present
verbatim. Nothing disputed is used as an input: NOT the anchor's
launch_to_memory_crossover_n, NOT its measured_memory_ceiling_GBps. The
ceiling is swept over a stated band instead, and the dependence of the
conclusion on that choice is reported.

CONVENTIONS, stated because they are the difference between a number and a
guess (the same lesson as 6.6.2's bytes_moved):
  k_empty      no bytes, no flops        -- pure launch cost
  k_noop_store reads 0, writes 0 (1 element touched, ignored as O(1))
  k_saxpy      reads 2 arrays, writes 1  -> 3 * n * 4 bytes, 2n flops
  k_relu       reads 1, writes 1         -> 2 * n * 4 bytes, 1 flop/elem
  k_copy       reads 1, writes 1         -> 2 * n * 4 bytes, 0 flops
  k_strided_read  reads n/stride, writes n -> (n/stride + n) * 4 bytes
  k_scatter_write reads 1 (+1 index), writes 1 -> (2n + 4n) bytes
  k_random_access  reads 1 (+1 seed), writes 1 -> (2n + 4n) bytes
  k_block_sum  reads 1, writes 1 per block -> 2 * n * 4 bytes
  k_tiled_gemm 2*n^3 flops, 3 * n^2 * 4 bytes
Bytes are counted for the elements the kernel's INDEXING actually touches,
which for the access-pattern family is not n -- the same per-kernel accounting
6.6.2 converged on.
"""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ANCHOR = HERE / "anchor_metrics.json"

# --- the 5090's own numbers, used ONLY as the model under test -------------
PEAK_FLOPS = 104.8e12          # fp32 CUDA-core peak, marketing figure
CEILING_BAND = {                # GB/s: stated alternatives, none of them
    "anchor_argmax": 1538.7,    #   taken on faith from the anchor
    "spec_1790": 1790.0,        #   the part's spec bandwidth
    "disputed_1541": 1541.3,    #   the value under dispute
    "low_1200": 1200.0,
}

MB = 1024 * 1024


def kernel_model(kernel: str, n: int, stride: int = 17):
    """(flops, bytes, convention) for one kernel instance of size n."""
    four = 4
    if kernel in ("k_empty", "k_noop_store"):
        return 0, 0, "no data touched (O(1)); launch cost only"
    if kernel == "k_saxpy":
        return 2 * n, 3 * n * four, "reads 2 arrays, writes 1"
    if kernel in ("k_relu", "k_copy", "k_block_sum"):
        return n, 2 * n * four, "reads 1, writes 1"
    if kernel == "k_strided_read":
        touched = max(1, n // stride)
        return n, (touched + n) * four, "reads n/stride, writes n"
    if kernel in ("k_scatter_write", "k_random_access"):
        return n, 2 * n * four + n * four, "reads 1 + 1 index/seed, writes 1"
    if kernel == "k_tiled_gemm":
        # problem_size is D*D (the flat buffer the driver allocates), NOT the
        # matrix dim: using 2*n^3 with n = D^2 gave a 137-second prediction
        # against a measured 1.8 ms, i.e. the same two-conventions trap 6.6.2
        # found in the byte accounting, here in the FLOP accounting.
        D = max(1, int(math.isqrt(n)))
        return 2 * (D ** 3), 3 * (D ** 2) * four, (
            f"2*D^3 flops and 3*D^2*4 bytes with D=sqrt(problem_size)={D}")
    raise KeyError(kernel)


def launch_table(cells):
    """Measured empty-kernel cost per (grid, block), straight from the anchor."""
    tbl = {}
    for c in cells:
        if c["kernel"] == "k_empty":
            tbl[(tuple(c["grid"]), tuple(c["block"]))] = c["median_ms"] * 1000.0
    return tbl


def fit_launch(tbl):
    """Least-squares fit of empty-kernel cost against BLOCK COUNT.

    Read straight off the anchor's own k_empty cells: 1-4 blocks cost
    1.5-1.8 us, 256 blocks 1.58, 4096 blocks 5.06, 16384 blocks 9.15. Block
    count is the regressor that fits; total threads is not -- two cells at
    65536 threads cost 1.58 and 1.89 while two at 4.19M threads cost 5.06 and
    9.15, which tracks blocks and not threads. Fitting on threads gave an
    intercept of 1.4e6 us, which is how the wrong regressor announces itself.
    Uses only the anchor's own empty cells: no disputed value, no magic 1.68.
    """
    xs, ys = [], []
    for (g, b), v in tbl.items():
        xs.append(g[0] * g[1] * g[2])          # BLOCKS, not threads
        ys.append(v)
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    # intercept = mean(y) - slope * mean(x). Writing mx here instead of my is
    # a one-character slip that leaves the SLOPE exactly right -- so nothing
    # looks wrong -- and the intercept wrong by three orders of magnitude
    # (3465.897 us instead of ~1.66 us). It produced a "launch-corrected
    # under-predicts by 118,287%" verdict, which is the kind of number that
    # looks like a finding.
    return my - slope * mx, slope, n


def launch_for(fit, grid, block, tbl):
    intercept, slope, _ = fit
    blocks = grid[0] * grid[1] * grid[2]
    exact = tbl.get((tuple(grid), tuple(block)))
    pred = intercept + slope * blocks
    src = (f"exact empty cell (grid={grid[0]}, block={block[0]}, "
           f"{exact:.3f} us)" if exact is not None else
           f"fit a + b*blocks = {intercept:.3f} + {slope:.3e}*{blocks}")
    return pred, src


def main() -> int:
    d = json.loads(ANCHOR.read_text())
    cells = d["cells"]
    tbl = launch_table(cells)

    fit = fit_launch(tbl)
    print(f"launch fit: t_us = {fit[0]:.3f} + {fit[1]:.3e} * blocks "
          f"({fit[2]} empty cells)")

    def launch_us(grid, block):
        return launch_for(fit, grid, block, tbl)

    rows = []
    for c in cells:
        n = c["n_problem_size"]
        flops, byts, conv = kernel_model(c["kernel"], n)
        if c["kernel"] == "k_empty":
            continue                      # the launch measurement itself
        t_launch, how = launch_us(c["grid"], c["block"])
        measured = c["median_ms"] * 1000.0
        entry = {
            "label": c["label"], "kernel": c["kernel"],
            "op_type": c["op_type"],
            "n_problem_size": n, "grid": c["grid"], "block": c["block"],
            "covers": c.get("covers"),
            "measured_median_us": measured,
            "flops": flops, "bytes": byts, "convention": conv,
            "t_launch_us": t_launch, "t_launch_source": how,
        }
        for name, ceiling in CEILING_BAND.items():
            t_flop = flops / PEAK_FLOPS * 1e6
            t_byte = byts / (ceiling * 1e9) * 1e6
            # The bead's DESIGN specifies min(flops/peak, bytes/ceiling). For a
            # fixed amount of work the kernel is bounded BELOW by BOTH
            # resources, so the time floor is max(...), not min(...): min
            # always selects the FLOP branch (which for a memory-bound kernel
            # is orders of magnitude smaller) and yields a prediction that is
            # ~100% low at EVERY size, including the largest. Both are
            # computed so the difference is visible rather than argued.
            t_min = min(t_flop, t_byte)
            t_max = max(t_flop, t_byte)
            entry[f"t_flop_bound_us_{name}"] = t_flop
            entry[f"t_byte_bound_us_{name}"] = t_byte
            entry[f"t_roofline_MIN_as_specified_us_{name}"] = t_min
            entry[f"t_roofline_MAX_correct_us_{name}"] = t_max
            entry[f"t_launch_corrected_us_{name}"] = t_launch + t_max
            entry["t_launch_corrected_with_MIN_us"] = t_launch + t_min
        rows.append(entry)

    def stats(keyfn):
        res = sorted(abs(r["measured_median_us"] - keyfn(r))
                      / r["measured_median_us"] * 100.0
                      for r in rows if r["measured_median_us"] > 0)
        return {"median_pct": statistics.median(res),
                "p90_pct": res[int(0.9 * len(res))],
                "max_pct": res[-1]}

    verdict = {}
    for ceiling in CEILING_BAND:
        verdict[ceiling] = {
            "roofline_MIN_as_specified": stats(
                lambda r, c=ceiling: r[f"t_roofline_MIN_as_specified_us_{c}"]),
            "roofline_MAX_correct": stats(
                lambda r, c=ceiling: r[f"t_roofline_MAX_correct_us_{c}"]),
            "launch_corrected": stats(
                lambda r, c=ceiling: r[f"t_launch_corrected_us_{c}"]),
            "launch_corrected_with_MIN_form": stats(
                lambda r: r["t_launch_corrected_with_MIN_us"]),
        }

    # the headline: the smallest n at which each form is usable, per SIZE BAND
    # rather than per n, because the anchor's n values are sparse and the
    # interesting question is "from which regime does the model work".
    BANDS = [(0, 1 << 12, "<=4K"), (1 << 12, 1 << 18, "4K-256K"),
             (1 << 18, 1 << 24, "256K-16M"), (1 << 24, 1 << 62, ">16M")]
    by_band = {}
    for lo, hi, name in BANDS:
        sub = [r for r in rows if lo <= r["n_problem_size"] < hi]
        if not sub:
            continue
        f_min = statistics.median(
            abs(r["measured_median_us"]
                - r["t_roofline_MIN_as_specified_us_anchor_argmax"])
            / r["measured_median_us"] for r in sub)
        f_max = statistics.median(
            abs(r["measured_median_us"]
                - r["t_roofline_MAX_correct_us_anchor_argmax"])
            / r["measured_median_us"] for r in sub)
        f_lc = statistics.median(
            abs(r["measured_median_us"]
                - r["t_launch_corrected_us_anchor_argmax"])
            / r["measured_median_us"] for r in sub)
        by_band[name] = {"n_cells": len(sub), "roofline_min_pct": f_min * 100,
                         "roofline_max_pct": f_max * 100,
                         "launch_corrected_pct": f_lc * 100}
        print(f"  band {name:10s} cells={len(sub):3d}  min={f_min*100:7.1f}%  "
              f"max={f_max*100:6.1f}%  +launch={f_lc*100:6.1f}%")
    plain_ok = next((n for n, v in by_band.items() if v["roofline_max_pct"] <= 10), None)
    launch_ok = next((n for n, v in by_band.items()
                      if v["launch_corrected_pct"] <= 10), None)

    # The anchor DELIBERATELY includes under-covered configs (1 block for a
    # 2^26 problem) -- 6.6.2 kept them because an anchor needs that dynamic
    # range. No roofline can predict those: there is no term for "your grid is
    # 65536x too small", and averaging them into a band median makes the model
    # look broken for the wrong reason. So the verdict separates cells whose
    # launch config actually COVERS the problem (threads >= n/4) from those
    # that do not.
    covered = [r for r in rows
               if r["grid"][0] * r["block"][0] >= r["n_problem_size"] / 4]
    covered_band = {}
    for lo, hi, name in BANDS:
        sub = [r for r in covered if lo <= r["n_problem_size"] < hi]
        if not sub:
            continue
        covered_band[name] = {
            "n_cells": len(sub),
            "roofline_MIN_pct": statistics.median(
                abs(r["measured_median_us"]
                    - r["t_roofline_MIN_as_specified_us_anchor_argmax"])
                / r["measured_median_us"] for r in sub) * 100,
            "roofline_MAX_pct": statistics.median(
                abs(r["measured_median_us"]
                    - r["t_roofline_MAX_correct_us_anchor_argmax"])
                / r["measured_median_us"] for r in sub) * 100,
            "launch_corrected_pct": statistics.median(
                abs(r["measured_median_us"]
                    - r["t_launch_corrected_us_anchor_argmax"])
                / r["measured_median_us"] for r in sub) * 100,
        }
    # ceiling dependence ON THE REGIME WHERE THE MODEL WORKS
    ceiling_dependence = {}
    for name in CEILING_BAND:
        big = [r for r in covered if r["n_problem_size"] > (1 << 24)]
        if big:
            ceiling_dependence[name] = statistics.median(
                abs(r["measured_median_us"]
                    - r["t_roofline_MAX_correct_us_" + name])
                / r["measured_median_us"] for r in big) * 100

    out = {
        "bead": "gpu-sim-u2o.6.6.4",
        "what": ("does an analytic roofline predict the anchor's measurements, "
                 "and with an explicit launch term added, where does each fail"),
        "inputs_used": {
            "anchor": str(ANCHOR),
            "anchor_cells": len(cells),
            "NOT_used": ["launch_to_memory_crossover_n", "measured_memory_ceiling_GBps"],
            "why_not": ("both are under dispute; the ceiling is swept over a "
                        "stated band instead and the dependence is reported"),
        },
        "model": {
            "peak_flops_fp32": PEAK_FLOPS,
            "ceiling_band_GBps": CEILING_BAND,
            "t_roofline": "min(flops/peak, bytes/ceiling)",
            "t_launch_corrected": "t_launch(grid, block) + t_roofline",
            "t_launch_source": ("measured from the anchor's OWN k_empty cells at "
                                "the same launch config, or the nearest one"),
        },
        "verdict_summary": {
            "well_covered_by_size_band": covered_band,
            "ceiling_dependence_on_covered_gt_16M": ceiling_dependence,
            "n_well_covered": len(covered), "n_total_cells": len(rows),
        },
        "per_cell": rows,
        "verdict": {
            "residual_by_form_and_ceiling": verdict,
            "residual_by_size_band": by_band,
            "smallest_band_plain_roofline_MAX_within_10pct": plain_ok,
            "smallest_band_launch_corrected_within_10pct": launch_ok,
            "informative_regime_decision_survives": bool(
                launch_ok is not None and (plain_ok is None or plain_ok > launch_ok)),
            "design_correction": (
                "the bead's DESIGN specifies t_roofline = min(flops/peak, "
                "bytes/ceiling). For a fixed amount of work the kernel is "
                "bounded BELOW by both resources, so the time floor is "
                "max(...). min() always selects the FLOP branch, which for a "
                "memory-bound kernel is orders of magnitude smaller, and "
                "yields a prediction ~100% low at EVERY size including the "
                "largest. Both forms are computed per cell so the difference "
                "is auditable rather than argued."),
        },
    }
    (HERE / "metrics.json").write_text(json.dumps(out, indent=2))
    print("=== residual (median % of measured) by form and ceiling ===")
    for c, v in verdict.items():
        print(f"  {c:16s} roofline[min, as specified]="
              f"{v['roofline_MIN_as_specified']['median_pct']:7.1f}%  "
              f"roofline[max, correct]={v['roofline_MAX_correct']['median_pct']:7.1f}%  "
              f"+launch={v['launch_corrected']['median_pct']:7.1f}%")
    print(f"\nplain roofline within 10% from n = {plain_ok}")
    print(f"launch-corrected within 10% from n = {launch_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
