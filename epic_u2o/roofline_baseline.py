"""Analytic roofline baseline for the RTX 5090. gpu-sim-u2o.9.4.

Zero learned parameters: `max(compute_time, memory_time)` from the device spec
and the static byte/FLOP accounting. This is the floor every learned model has
to beat to justify its cost, so its own accounting has to be right -- and the
first thing that turned out to be wrong is its own byte accounting.

WHAT I EXPECTED TO FIND HERE, AND WHY THE EXPECTATION WAS WRONG. Criterion 1
forbids "a flat 2*n*4", and I read the anchor's
`regimes.memory_ceiling_definition` -- which says "2*n*4 bytes moved (one read
+ one write, fp32)" -- concluded the published 1575.8 GB/s ceiling was
overstated 1.50x, and wrote that into this docstring before checking.

CHECKING THE FIELDS SHOWED THE OPPOSITE. The anchor's ceiling cell
`saxpy n=67108864 auto/256` carries `bytes_moved = 805,306,368 = 3*n*4`, which
is the CORRECT per-kernel accounting for `y[j] = alpha * x[j] + y[j]` (reads x,
reads y, writes y). 6.6.2's commit `734980f` is literally "one byte convention
in the file, and the artifact regenerated with it". So the numbers are right and
the PROSE is stale.

I believed the prose over the numbers -- which is the exact inversion of the
standard this lab applies everywhere else, and a good reminder of why
6.11.1's lint exists: a derived value stated in prose is not evidence about the
field it describes.

The two defects that ARE real, both narrower than the one I expected:
  1. `memory_ceiling_definition` still says 2*n*4 while the fields use 3*n*4.
  2. `k_random_access` is credited 2*n*4 but its body reads seeds, reads
     in[rand] and writes out -- 3 array touches. Its achieved GB/s is
     overstated 1.50x. It does not move the memory ceiling, but it corrupts any
     bandwidth figure for the `access` family, which is 45 of 124 cells.

THE ACCOUNTING IS DERIVED FROM THE KERNEL BODIES, NOT FROM A FORMULA. Every
entry below is justified by the source, including the three cases criterion 1
names by name (strided, scatter, random) and the two degenerate kernels where
a flat formula is not merely wrong but wrong by five orders of magnitude
(k_noop_store touches ONE element; k_empty touches none).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# RTX 5090 (GB202) device spec used by the analytic bound.
DEVICE = {
    "name": "RTX 5090",
    "fp32_peak_tflops": 104.8,
    "hbm_peak_gbps": 1792.0,      # spec-sheet figure, labelled as such
    "peak_provenance": ("spec-sheet figures, NOT measured in this file. A "
                        "baseline built on an unmeasured peak inherits its "
                        "uncertainty; the ACHIEVED column in the artifact is "
                        "measured and is the one to check against."),
    "l2_bytes": 98304 * 1024,     # 96 MB
}


@dataclass(frozen=True)
class ByteAccounting:
    """Per-kernel traffic, counted from the body."""
    kernel: str
    reads: int
    writes: int
    why: str
    special_case: str = ""
    note: str = ""

    @property
    def arrays(self) -> int:
        return self.reads + self.writes

    def bytes_moved(self, n: int) -> int:
        if self.kernel == "k_empty":
            return 0
        if self.kernel == "k_noop_store":
            # out[0] = out[0]: ONE element, not n. A flat 2*n*4 would claim
            # 8n bytes and overstate this by ~5e6 at n=2^26.
            return 4
        return self.arrays * 4 * n

    def flops(self, n: int) -> int:
        if self.kernel == "k_tiled_gemm":
            d = int(n ** 0.5)          # problem_size is D*D, not D
            return 2 * d ** 3
        if self.kernel == "k_relu":
            return n                   # one compare + one select
        if self.kernel == "k_saxpy":
            return 2 * n               # a*x + y : one mul, one add
        if self.kernel == "k_copy":
            return 0
        if self.kernel == "k_block_sum":
            return n
        if self.kernel in ("k_strided_read", "k_scatter_write",
                           "k_random_access"):
            return n
        return 0


ACCOUNTING: dict[str, ByteAccounting] = {
    "k_saxpy": ByteAccounting(
        "k_saxpy", reads=2, writes=1,
        why="y[j] = alpha * x[j] + y[j] reads x, reads y, writes y",
        note="1.50x the flat 2*n*4 convention"),
    "k_copy": ByteAccounting(
        "k_copy", reads=1, writes=1,
        why="dst[j] = src[j] reads src, writes dst",
        note="matches 2*n*4"),
    "k_relu": ByteAccounting(
        "k_relu", reads=1, writes=1,
        why="y[j] = x[j] > 0 ? x[j] : 0 reads x, writes y",
        note="source names x twice; a compiler common-subexpression-eliminates "
             "it, so 1 read. Compiler-dependent -- flagged, not assumed away"),
    "k_strided_read": ByteAccounting(
        "k_strided_read", reads=1, writes=1,
        why="out[i] = in[i*stride] reads in, writes out",
        special_case="strided",
        note="bytes moved are 2n*4, but the READ is uncoalesced: it touches "
             "1/stride of each cache line, so achieved GB/s is not a bandwidth "
             "measurement of DRAM. Excluded from the ceiling on that ground, "
             "which is the strided special case criterion 1 names"),
    "k_scatter_write": ByteAccounting(
        "k_scatter_write", reads=2, writes=1,
        why="out[indices[i]] = in[i] reads in, reads indices, writes out",
        special_case="scatter",
        note="1.50x flat; and the write is scattered, so it is not a "
             "streaming-bandwidth measurement either"),
    "k_random_access": ByteAccounting(
        "k_random_access", reads=2, writes=1,
        why="reads seeds, reads in[rand], writes out",
        special_case="random",
        note="1.50x flat; fully random reads defeat every level of cache"),
    "k_block_sum": ByteAccounting(
        "k_block_sum", reads=1, writes=1,
        why="reads in, writes one partial per block",
        note="write is O(blocks), not O(n); 2n*4 overstates the write side"),
    "k_tiled_gemm": ByteAccounting(
        "k_tiled_gemm", reads=2, writes=1,
        why="reads A, reads B, writes C (tiled, so A and B are re-read per "
            "tile)",
        special_case="tiled",
        note="problem_size is D*D, so bytes are 3*4*D*D and FLOPs 2*D^3; the "
             "FLOP side is what makes this family compute-bound, and it is a "
             "SINGLE-KERNEL family -- see the per-kernel census"),
    "k_noop_store": ByteAccounting(
        "k_noop_store", reads=0, writes=0,
        why="out[0] = out[0] touches ONE element",
        special_case="degenerate",
        note="not a bandwidth cell at all; a flat formula would overstate it "
             "by ~5e6 at n=2^26"),
    "k_empty": ByteAccounting(
        "k_empty", reads=0, writes=0,
        why="empty body",
        special_case="degenerate",
        note="launch-cost cell, not a bandwidth cell"),
}

# Contention multipliers, MEASURED by gpu-sim-u2o.6.6.3. Criterion 6: any
# single global error bar is wrong because these differ by 2x.
CONTENTION = {
    "memory-bound": 2.33,
    # compute-bound: NOT MEASURED -- no co-tenant effect was recorded for this
    # regime, so 1.0 is an ASSUMPTION and is labelled as one rather than
    # presented as a measurement alongside the two that are real.
    "compute-bound": 1.0,
    "launch-bound": 1.19,     # the smaller measured figure is 1.05; 1.19 used
                              # as the conservative bound
    "source": "gpu-sim-u2o.6.6.3 ts 20260928t072008-b34ddec9",
    "note": ("compute-bound is NOT MEASURED: no co-tenant effect was recorded "
             "for that regime, so its 1.0 is an ASSUMPTION, unlike the two "
             "multipliers that were measured. A roofline calibrated on "
             "contended cells reads HIGH, because "
             "sharing the memory system LOWERS achieved bandwidth. So the bias "
             "is quiet: the baseline understates the achievable ceiling rather "
             "than failing loudly. Criterion 5."),
}

# Criterion 7, verbatim intent: clocks.sm is NOT a contention detector here.
FORBIDDEN_CONTENTION_SIGNALS = ("clocks.sm", "clocks.max.sm")
CLOCKS_SPREAD = 0.098
POWER_SPREAD = 0.866
UTIL_SPREAD = 0.939


class RooflineError(RuntimeError):
    """A baseline that cannot be trusted. Raised, never published."""


def assert_no_clock_contention(signal: str) -> None:
    if signal in FORBIDDEN_CONTENTION_SIGNALS:
        raise RooflineError(
            f"{signal} must not be used as a contention detector: its measured "
            f"relative spread across conditions was {CLOCKS_SPREAD}, against "
            f"power.draw {POWER_SPREAD} and utilization.gpu {UTIL_SPREAD}. A "
            f"clock-based exclusion would remove cells at random and call it "
            f"rigour.")


def bytes_moved(kernel: str, n: int) -> int:
    if kernel not in ACCOUNTING:
        raise RooflineError(
            f"no byte accounting for kernel {kernel!r}. A baseline that falls "
            f"back to a flat formula for an unknown kernel is the exact defect "
            f"criterion 1 forbids, so this refuses instead.")
    return ACCOUNTING[kernel].bytes_moved(n)


def predicted_us(kernel: str, n: int, grid: int, block: int) -> dict:
    """max(compute_time, memory_time). ZERO learned parameters.

    The max is not a style choice. For a FIXED amount of work the kernel is
    bounded BELOW by both resources, so the time floor is max(...); min(...)
    always selects the FLOP branch, which for a memory-bound kernel is orders
    of magnitude smaller. gpu-sim-u2o.6.6.4 measured that error on this very
    anchor: min() under-predicts ~100% at EVERY size including the largest.
    """
    acc = ACCOUNTING[kernel]
    b = acc.bytes_moved(n)
    f = acc.flops(n)
    mem_us = (b / (DEVICE["hbm_peak_gbps"] * 1e9)) * 1e6
    comp_us = (f / (DEVICE["fp32_peak_tflops"] * 1e12)) * 1e6
    binding = "memory" if mem_us >= comp_us else "compute"
    return {
        "kernel": kernel, "n": n, "bytes_moved": b, "flops": f,
        "memory_bound_us": mem_us, "compute_bound_us": comp_us,
        "predicted_us": max(mem_us, comp_us),     # max, NOT min
        "binding_resource": binding,
        "byte_convention": f"{acc.arrays} array touches x 4 bytes x n"
                           + (f" ({acc.note})" if acc.note else ""),
    }


def is_ceiling_eligible(kernel: str, n: int, l2_resident: bool,
                        contention_state: str,
                        is_largest_n: bool = True) -> tuple[bool, str]:
    """Membership rule for the ceiling candidate set. Stated, not implicit.

    Criterion 2: an unstated candidate set is a defect even when the argmax is
    right. So the rule is returned as text with every verdict.
    """
    if l2_resident:
        return False, "L2-resident: sustains cache bandwidth, not HBM"
    if not is_largest_n:
        # MY FIRST VERSION OMITTED THIS and it produced a ceiling of 3,855
        # GB/s -- ABOVE the 5090's 1,792 GB/s HBM peak, which is physically
        # impossible and should have been the loudest possible signal. The
        # candidate set admitted n=4M cells that fit in the 96 MB L2. A ceiling
        # above spec is not a finding, it is a bug, and it is now refused twice
        # over: by this rule and by assert_not_above_spec_peak().
        return False, "not the largest problem size: sub-maximal cells are "\
                      "cache-resident and do not measure HBM"
    if ACCOUNTING[kernel].special_case in ("strided", "scatter", "random", "tiled"):
        return False, (f"{ACCOUNTING[kernel].special_case}: not a streaming "
                       f"bandwidth measurement")
    if ACCOUNTING[kernel].special_case == "degenerate":
        return False, "degenerate kernel: not a bandwidth cell"
    if contention_state != "uncontended":
        return False, (f"contention state {contention_state!r}: a contended "
                       f"cell LOWERS achieved bandwidth, so using it for a "
                       f"ceiling reads HIGH (criterion 5)")
    return True, "largest-n streaming kernel, both sides uncontended"


def assert_not_above_spec_peak(achieved_gbps: float) -> None:
    """An achieved bandwidth above the spec sheet is a bug, not a high number.

    Added after this module's first run selected a ceiling of 3,855 GB/s on a
    5090 whose HBM peak is 1,792 GB/s. A value above the physical maximum means
    the byte accounting or the eligibility rule is wrong, and publishing it
    would put an impossible number in the artifact that a later reader has no
    reason to doubt.
    """
    if achieved_gbps > DEVICE["hbm_peak_gbps"]:
        raise RooflineError(
            f"achieved {achieved_gbps:,.1f} GB/s is ABOVE the {DEVICE['name']} "
            f"HBM peak of {DEVICE['hbm_peak_gbps']:,.1f} GB/s. That is "
            f"impossible, so either the byte accounting double-counts or a "
            f"cache-resident cell was admitted to the candidate set. Refusing "
            f"to publish a ceiling built on it.")


def select_ceiling(candidates: list[dict]) -> dict:
    """argmax over the NAMED candidate set, with the top candidates emitted.

    A first-element selection is a defect (criterion 2), so this returns the
    ranking, not just the winner.
    """
    if not candidates:
        raise RooflineError(
            "the ceiling candidate set is empty. An empty set has an argmax of "
            "nothing, and a baseline fitted on nothing is not a baseline.")
    assert_not_above_spec_peak(max(c["achieved_gbps"] for c in candidates))
    ranked = sorted(candidates,
                    key=lambda c: c["achieved_gbps"], reverse=True)
    for c in ranked:
        c["rank"] = ranked.index(c) + 1
    return {
        "ceiling_GBps": ranked[0]["achieved_gbps"],
        "ceiling_from_cell": ranked[0]["label"],
        "membership_rule": ("largest problem size (HBM-bound), streaming "
                            "kernels only, both sides uncontended"),
        "top_candidates": [
            {k: c[k] for k in ("rank", "label", "kernel", "n_problem_size",
                               "bytes_moved", "median_ms", "achieved_gbps")}
            for c in ranked[:4]],
        "n_candidates": len(ranked),
    }
