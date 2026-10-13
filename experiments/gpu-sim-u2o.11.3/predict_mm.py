#!/usr/bin/env python3
"""u2o.11.3 regime extension: predict memory->memory cells BEFORE generating.

The 1693-row corpus contains zero memory->memory cells under 9.4's
construction (census FAIL). Probing the construction showed the flip needs
~100MB+ counted bytes at modest block counts: high-ref-count ops
(gather/scatter/random, reductions, shared-memory patterns, stencil,
histograms) at large n with full-coverage grids. 2-ref streaming never
flips below 64M at any grid (even 131072 blocks), so the extension is
chosen for REGIME COVERAGE, not row count -- the 6.7.1 precedent
(deliberate grid fractions correcting a 73% launch-bound first pass).

This script classifies exact candidate (op, n, grid, block) cells through
the same code path as census.py (live 8.3 features into the two pinned
historical rooflines) and writes the predicted-flip keep-list. build_batch
generates EXACTLY these cells -- nothing unmeasured, nothing unpredicted.
All cells use full coverage (threads == n): covers is maximal (bytes count
fully) and every OOB guard in the templates holds with equality.

Shared memory MUST equal the construction values (not zero): occupancy
feeds both pinned models, so predicting with shmem=0 then measuring with
shmem=block*4 could flip the class after the fact. SHMEM_1D records the
generators' own get_shared_mem rules; the 2D cells carry theirs.

Validity notes (checked against templates, recorded here):
- gather/scatter indices and random seeds are all < n by fill/LCG-mod.
- shared_mem_bank_conflict_* writes out[block*blockDim+tid] with NO i<n
  guard: full coverage (threads == n) is exactly safe; over-coverage
  would crash. The keep-list enforces threads == n for every cell.
- reduce_sum_warp_shuffle atomicAdds out[0]: one per warp, always safe.
- 2D stencil/transpose cells carry rows/cols metadata (driver dims rule);
  grids are exact full coverage.
- histogram_shared needs extern shared = nbins*4 bytes: the keep-list
  records shared_mem_bytes (nbins=1024 -> 4096B); the driver must pass it.
  histogram bins are safe: in[] fills cast to non-negative ints, b%nbin.
- 64M cells: 768MB host buffers fit the 16GB device; host fill ~2s/row is
  CPU-side (extension jobs stay under the 5-minute underuse grace: the
  gate only fires after grace expiry).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from epic_u2o.features import LaunchConfig, extract_features  # noqa: E402
from epic_u2o.generator_stage2 import generate_kernels as g2  # noqa: E402
from epic_u2o.generator_stage3 import generate_kernels as g3  # noqa: E402
from epic_u2o.generator_stage5 import generate_kernels as g5  # noqa: E402
from census import classify  # noqa: E402

SOURCES = {}
for gen in (g2, g3, g5):
    for s in gen(max_kernels=100000):
        SOURCES.setdefault(s.op_type, s.to_dict()["source"])

OPS_1D = ["gather_read", "random_access", "scatter_write",
          "reduce_max_block", "reduce_min_block", "reduce_sum_block",
          "reduce_sum_warp_shuffle",
          "shared_mem_bank_conflict_1way", "shared_mem_bank_conflict_2way",
          "shared_mem_bank_conflict_32way", "shared_mem_reduction_tiled"]
# Construction shared memory per op (the generators' get_shared_mem rules).
# reduce/tiled need block*4 (extern sdata[tid]); bank-conflict tiles are
# fixed (1way 32x33, 2way 32x34, 32way 32x1 floats).
SHMEM_1D = {"reduce_max_block": lambda bb: bb * 4,
            "reduce_min_block": lambda bb: bb * 4,
            "reduce_sum_block": lambda bb: bb * 4,
            "shared_mem_reduction_tiled": lambda bb: bb * 4,
            "shared_mem_bank_conflict_1way": lambda bb: 32 * 33 * 4,
            "shared_mem_bank_conflict_2way": lambda bb: 32 * 34 * 4,
            "shared_mem_bank_conflict_32way": lambda bb: 32 * 4}
# full-coverage (grid x block == n) grids, smallest-block-count first.
GRIDS = {2097152: [(2048, 1024), (4096, 512), (8192, 256)],
         4194304: [(4096, 1024), (8192, 512), (16384, 256)],
         8388608: [(8192, 1024), (16384, 512), (32768, 256)],
         16777216: [(16384, 1024), (32768, 512), (65536, 256)],
         33554432: [(32768, 1024), (65536, 512), (131072, 256)],
         67108864: [(65536, 1024), (131072, 512), (262144, 256)]}
SIZES_HIREF = [2097152, 4194304, 8388608, 16777216, 33554432, 67108864]
SIZES_HIST = [2097152, 4194304, 8388608, 16777216, 33554432]
# 2D cells: (op, (rows, cols), (bx, by), shmem) full coverage exact.
# stencil shmem 34*34*4 and transpose 32*32*4 are the generators' rules.
CELLS_2D = [(("stencil_2d_5pt", (1024, 1024), (16, 16), 34 * 34 * 4)),
            (("stencil_2d_5pt", (1024, 1024), (32, 32), 34 * 34 * 4)),
            (("stencil_2d_5pt", (1024, 2048), (16, 16), 34 * 34 * 4)),
            (("stencil_2d_5pt", (1024, 2048), (32, 32), 34 * 34 * 4)),
            (("stencil_2d_5pt", (2048, 2048), (16, 16), 34 * 34 * 4)),
            (("stencil_2d_5pt", (2048, 2048), (32, 32), 34 * 34 * 4)),
            (("stencil_2d_5pt", (2048, 4096), (32, 32), 34 * 34 * 4)),
            (("stencil_2d_5pt", (4096, 4096), (32, 32), 34 * 34 * 4)),
            (("shared_mem_transpose", (2048, 2048), (32, 32), 32 * 32 * 4)),
            (("shared_mem_transpose", (2048, 4096), (32, 32), 32 * 32 * 4)),
            (("shared_mem_transpose", (4096, 4096), (32, 32), 32 * 32 * 4))]


def cell_1d(op, n, gg, bb, shmem=0, extra=None):
    assert gg * bb == n, (op, n, gg, bb)
    lc = LaunchConfig(grid_x=gg, grid_y=1, grid_z=1,
                      block_x=bb, block_y=1, block_z=1,
                      shared_mem_bytes=shmem)
    f = extract_features(SOURCES[op], lc, n)
    if classify(f) == "memory->memory":
        cell = {"op": op, "n": n, "grid": [gg, 1, 1], "block": [bb, 1, 1],
                "shared_mem_bytes": shmem,
                "bytes_moved": f.estimated_bytes_moved}
        if extra:
            cell.update(extra)
        return cell
    return None


def main() -> int:
    keep = []
    for op in OPS_1D:
        for n in SIZES_HIREF:
            for (gg, bb) in GRIDS[n]:
                shmem = SHMEM_1D.get(op, lambda bb: 0)(bb)
                k = cell_1d(op, n, gg, bb, shmem)
                if k:
                    keep.append(k)
    for op in ["histogram_global", "histogram_shared"]:
        for n in SIZES_HIST:
            for (gg, bb) in GRIDS[n]:
                shmem = 4096 if op == "histogram_shared" else 0
                k = cell_1d(op, n, gg, bb, shmem,
                             {"nbins": 1024})
                if k:
                    keep.append(k)
    for (op, (rows, cols), (bx, by), shmem) in CELLS_2D:
        n = rows * cols
        gx, gy = cols // bx, rows // by
        assert gx * bx == cols and gy * by == rows
        lc = LaunchConfig(grid_x=gx, grid_y=gy, grid_z=1,
                          block_x=bx, block_y=by, block_z=1,
                          shared_mem_bytes=shmem)
        f = extract_features(SOURCES[op], lc, n)
        if classify(f) == "memory->memory":
            keep.append({"op": op, "n": n,
                         "grid": [gx, gy, 1], "block": [bx, by, 1],
                         "rows": rows, "cols": cols,
                         "shared_mem_bytes": shmem,
                         "bytes_moved": f.estimated_bytes_moved})
    (HERE / "mm_keeplist.json").write_text(json.dumps(keep, indent=2))
    from collections import Counter
    print(f"predicted memory->memory: {len(keep)} cells")
    print(Counter((k["op"], k["n"]) for k in keep))
    if len(keep) < 105:
        print("KEEP-LIST SHORT: fewer than 105 flips -- do not generate; "
              "extend candidates first")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
