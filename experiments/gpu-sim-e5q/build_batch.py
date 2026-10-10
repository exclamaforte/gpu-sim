#!/usr/bin/env python3
"""gpu-sim-e5q EXTENSION (CPU): build the large-size manifest.

The bead's audit note is the spec: generator_stage1.py runs and emits
runs/stage1_dataset/*.jsonl with 1000 rows, but it has NO runtime column, so
only the generate step of generate -> compile -> preflight -> measure ->
tokenize exists. This script produces the input half of the missing half, with
the ranges chosen from what the lab has since measured rather than guessed:

  * gpu-sim-u2o.6.6.1 measured that at n <= 2^18 the stage-1 kernels run in
    2-3 us, which is BELOW the ~1.3 us irreducible dispatch cost plus the
    single-launch measurement overhead: those cells are majority launcher, and
    their across-invocation CV is ~7%. So the small end of the sweep is kept
    for coverage but is LABELLED, and the bulk of the batch is placed in the
    >= 10 us regime where 6.6.1 measured a ~0.13% across-invocation CV.
  * gpu-sim-u2o.6.6.3 measured a co-tenant puts up to 2.33x on a memory-bound
    cell and 1.19x on a sub-10 us one, so the batch records queue state and
    telemetry per row rather than assuming a quiet box.
  * The deliberately poor configs (1 block, 1 thread) are INCLUDED on
    purpose: 6.6.2's anchor needed that dynamic range to be an anchor, and a
    dataset that only contains the best config teaches a model nothing about
    the launch-config axis.

CPU only, no GPU, no queue. Writes manifest.json + one .cu per spec.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.generator_stage1 import generate_kernels  # noqa: E402
from epic_u2o.kernel_names import declared_kernel_names  # noqa: E402

# The sweep. Grids are DERIVED from the problem size rather than fixed,
# because a fixed small grid against a large n is majority launcher by
# construction: a first pass with fixed grids produced 264 of 360 cells
# launch-bound (73%), which is exactly the regime 6.6.1 showed is dominated
# by dispatch. So for each (n, block) we emit a full-coverage grid plus two
# partial ones (1/8 and 1/2 of full), which keeps real launch-config
# variation without filling the batch with cells that measure the launcher.
BLOCKS = [[64, 1, 1], [128, 1, 1], [256, 1, 1], [1024, 1, 1]]
# gpu-sim-e5q: ADDITIVE sizes only. The 2^12-2^18 rows already exist in
# u2o.7.1's measured.jsonl and are NOT regenerated (acceptance 2).
SIZES = [1 << 22, 1 << 23, 1 << 24]


def grids_for(n: int, block: int) -> list:
    """Full coverage, plus 1/8 and 1/2 of it, plus the poor 1-block case."""
    block_threads = block[0] * block[1] * block[2]
    full = max(1, n // block_threads)
    out = [[full, 1, 1], [max(1, full // 2), 1, 1], [max(1, full // 8), 1, 1]]
    if full > 64:                      # the deliberate under-coverage case
        out.append([1, 1, 1])
    return out
OPS = ["scale", "add", "saxpy", "relu", "copy", "triad", "mul", "div"]
# Below this, 6.6.1 measured the cell to be launch-dominated. Kept, labelled.
LAUNCH_BOUND_N = 1 << 14


def main() -> int:
    out = HERE / "build"
    out.mkdir(exist_ok=True)
    specs = []
    for n in SIZES:
        for block in BLOCKS:
            for grid in grids_for(n, block):
                for spec in generate_kernels(ops=OPS, grid_sweeps=[grid],
                                             block_sweeps=[block],
                                             problem_sizes=[n]):
                    specs.append(_row(spec, n, block, grid))
    manifest = HERE / "build" / "manifest.json"
    manifest.write_text(json.dumps(
        {"bead": "gpu-sim-e5q", "stage": "build",
         "n_specs": len(specs),
         "ops": OPS, "blocks": BLOCKS, "sizes": SIZES,
         "grid_rule": "full coverage + 1/2 + 1/8 of full, plus a 1-block case",
         "launch_bound_n_threshold": LAUNCH_BOUND_N,
         "note": ("grids derived from n: a fixed small grid against a large "
                  "n is majority launcher by construction, and 6.6.1 showed "
                  "those cells are dominated by dispatch. The 1-block case is "
                  "kept and flagged deliberately_poor -- an anchor needs that "
                  "dynamic range."),
         "specs": specs}, indent=2))
    print(f"built {len(specs)} specs -> {manifest}")
    poor = sum(1 for s in specs if s["deliberately_poor"])
    print(f"  deliberately_poor: {poor}")
    return 0


def _row(spec, n, block, grid) -> dict:
    """One manifest row, with the gpu-sim-9ti guard and the 6.6.1 labels."""
    out = HERE / "build"
    d = spec.to_dict()
    # gpu-sim-9ti: never trust kernel_name against source again.
    names = declared_kernel_names(d["source"])
    if d["kernel_name"] not in names:
        raise ValueError(
            f"kernel_name={d['kernel_name']!r} is not among the __global__ "
            f"functions in its own source {names} (gpu-sim-9ti)")
    threads = (d["grid"][0] * d["grid"][1] * d["grid"][2]
               * d["block"][0] * d["block"][1] * d["block"][2])
    d["covers"] = min(threads, d["problem_size"])
    # gpu-sim-u2o.7.1 rework: regime_hint REMOVED. The design-time guess was
    # wrong wherever it mattered (128/264 rows said measurable_expected for
    # launch_bound measurements); measured_regime is authoritative and the
    # only regime field shipped.
    d["deliberately_poor"] = (d["grid"] == [1, 1, 1] or d["block"] == [1, 1, 1])
    d["kernel_hash"] = hashlib.sha256(
        (d["source"] + json.dumps([d["grid"], d["block"],
                                   d["problem_size"]])).encode()).hexdigest()[:16]
    cu = out / f"k_{d['kernel_hash']}.cu"
    cu.write_text(d["source"])
    d["source_path"] = cu.name
    return d


if __name__ == "__main__":
    raise SystemExit(main())
