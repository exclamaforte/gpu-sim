#!/usr/bin/env python3
"""gpu-sim-0vk PREDICT (CPU): keep-list for the low-noise extension, BEFORE generation.

Reuses 11.3's validated mechanism (predict_mm.py): candidate cells through
census.py's classify() on live 8.3 features, keep-list recorded with the
prediction, gate refuses to generate below threshold. Two predicted
quantities per cell this time (0vk needs BOTH numbers for 9.1's revisit):

- census class via classify() (identical code path as the verdict);
- measurable forecast: the OLD (pre-46p, newer verdict model) pinned
  roofline predicted_time_ms >= 0.015. The 0.015 (not the 0.010 verdict
  line) is deliberate margin: on a 200-row 11.3 sample the 0.010 rule
  agrees with measured_regime 190/200, with misses concentrated at the
  boundary. The forecast is a screen, the measured med is the verdict.

GATE (acceptance 1): predicted memory->memory >= 280 AND predicted
measurable (conservative) >= 900, else exit 1 and nothing generates. The
margins cover forecast miss + the 257 mm / ~800 measurable gaps with room.

Novelty is part of the prediction: candidates colliding with any frozen
2278 hash are dropped here (build_extension asserts), so the predicted
counts are shippable rows, not occurrences. One scalar variant per
(op, n, grid, block) -- hash excludes scalars, so scalar variants would be
phantoms that inflate occurrence counts without adding rows.

Structural variety (item 4 / trap 2): high-ref-count families at large n
for mm (reductions, gather/scatter/random, bank-conflict, histograms,
stencil/transpose, GEMM), plus large streaming/atomic/divergent families
for measurable breadth. Plain elementwise-at-larger-n is capped: it adds
no new kernel structure.

All cells full coverage (threads == n): covers maximal, OOB guards hold
with equality (predict_mm validity notes). Shared memory = construction
values, never zero.
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "gpu-sim-u2o.11.3"))

from epic_u2o.features import LaunchConfig, extract_features  # noqa: E402
from epic_u2o.generator_stage1 import generate_kernels as g1  # noqa: E402
from epic_u2o.generator_stage2 import generate_kernels as g2  # noqa: E402
from epic_u2o.generator_stage3 import generate_kernels as g3  # noqa: E402
from epic_u2o.generator_stage4 import generate_kernels as g4  # noqa: E402
from epic_u2o.generator_stage5 import generate_kernels as g5  # noqa: E402
from epic_u2o.generator_stage1 import kernel_hash_for  # noqa: E402
from census import classify, _models  # noqa: E402 identical path (11.3)

SOURCES = {}
for gen in (g1, g2, g3, g4, g5):
    for s in gen(max_kernels=100000):
        SOURCES.setdefault(s.op_type, s.to_dict()["source"])

# (op, extra scalar metadata) families. Scalars fixed per cell (hash-novelty).
HIREF = [  # high-ref-count: the mm lever (11.3's OPS_1D + histograms)
    ("gather_read", {}), ("random_access", {}), ("scatter_write", {}),
    ("reduce_max_block", {}), ("reduce_min_block", {}),
    ("reduce_sum_block", {}), ("reduce_sum_warp_shuffle", {}),
    ("shared_mem_bank_conflict_1way", {}),
    ("shared_mem_bank_conflict_2way", {}),
    ("shared_mem_bank_conflict_32way", {}),
    ("shared_mem_reduction_tiled", {}),
    ("histogram_global", {"nbins": 1024}),
    ("histogram_shared", {"nbins": 1024}),
]
STREAM = [  # large streaming: measurable breadth, NOT the mm lever (capped)
    ("coalesced_read", {}), ("coalesced_write", {}),
    ("strided_read", {"stride": 8}),
    ("strided_read_write", {"stride": 8}),
    ("strided_write", {"stride": 8}),
    ("interleaved_read", {"interleave": 4}),
    ("block_cyclic", {"block_size": 1024}),
    ("segmented_read", {"segment_size": 4096}),
]
ATOMIC_DIV = [  # serialize/diverge at scale: measurable, family breadth
    ("atomic_counter", {}), ("divergent_branch", {}),
    ("divergent_loop", {}), ("divergent_switch", {}),
]
SHMEM_1D = {"reduce_max_block": lambda bb: bb * 4,
            "reduce_min_block": lambda bb: bb * 4,
            "reduce_sum_block": lambda bb: bb * 4,
            "shared_mem_reduction_tiled": lambda bb: bb * 4,
            "shared_mem_bank_conflict_1way": lambda bb: 32 * 33 * 4,
            "shared_mem_bank_conflict_2way": lambda bb: 32 * 34 * 4,
            "shared_mem_bank_conflict_32way": lambda bb: 32 * 4,
            "histogram_shared": lambda bb: 4096}
# Size tiers: powers of two PLUS 3x and 5x multiples (3M = 3072x1024,
# 5M = 5120x1024, so bb=1024 full coverage stays exact). mmx+base ate the
# plain-pow2 2M-64M space; these n are all novel hashes. Cap 768M floats
# (3GB/buffer, 9GB for 3-buffer ops) -- fits 16GB with headroom.
def _tiers(ks):
    return sorted({m * (2 ** k) for m in (1, 3, 5) for k in ks})
SIZES_HIREF = _tiers(range(20, 30))      # 1M..512M x{1,3,5}, <=768M
SIZES_HIREF = [n for n in SIZES_HIREF if n <= 768 * 2**20]
SIZES_STREAM = [n for n in _tiers(range(23, 30)) if n <= 768 * 2**20]
SIZES_ATOMIC = [8388608, 16777216]
GRIDS_HIREF = [512, 1024]  # modest block counts flip; bb<=256 never did
# Plain elementwise at large n: measurable breadth ONLY, capped by design
# (3 sizes x 2 grids x 8 ops = 48 candidates max, ~5% of keep). Trap 2:
# this must not become the lever; the mm comes from HIREF/GEMM/2D.
ELEM1_OPS = ["add", "copy", "div", "mul", "relu", "saxpy", "scale", "triad"]
SIZES_ELEM1 = [33554432, 67108864, 134217728]
BLOCKS = [64, 128, 256, 512, 1024]  # full coverage: gg = n // bb
# 2D cells: (op, (rows, cols), (bx, by), shmem) full coverage exact.
CELLS_2D = [
    ("stencil_2d_5pt", (2048, 2048), (16, 16), 34 * 34 * 4),
    ("stencil_2d_5pt", (2048, 2048), (32, 32), 34 * 34 * 4),
    ("stencil_2d_5pt", (2048, 4096), (16, 16), 34 * 34 * 4),
    ("stencil_2d_5pt", (2048, 4096), (32, 32), 34 * 34 * 4),
    ("stencil_2d_5pt", (4096, 4096), (16, 16), 34 * 34 * 4),
    ("stencil_2d_5pt", (4096, 4096), (32, 32), 34 * 34 * 4),
    ("stencil_2d_5pt", (2048, 8192), (32, 32), 34 * 34 * 4),
    ("stencil_2d_5pt", (4096, 8192), (32, 32), 34 * 34 * 4),
    ("shared_mem_transpose", (4096, 8192), (32, 32), 32 * 32 * 4),
    ("shared_mem_transpose", (8192, 8192), (32, 32), 32 * 32 * 4),
    ("diagonal_access", (2048, 2048), (16, 16), 0),
    ("diagonal_access", (2048, 2048), (32, 32), 0),
    ("diagonal_access", (4096, 4096), (32, 32), 0),
    ("diagonal_access", (2048, 8192), (32, 32), 0),
    ("shared_mem_transpose", (2048, 2048), (32, 32), 32 * 32 * 4),
    ("shared_mem_transpose", (2048, 4096), (32, 32), 32 * 32 * 4),
    ("shared_mem_transpose", (4096, 4096), (16, 16), 32 * 32 * 4),
    ("shared_mem_transpose", (4096, 4096), (32, 32), 32 * 32 * 4),
    ("shared_mem_transpose", (2048, 8192), (32, 32), 32 * 32 * 4),
]
MEAS_MARGIN_MS = 0.015  # forecast line (verdict line is 0.010)
GATE_MM = 280
GATE_MEAS = 900


def frozen_hashes():
    out = set()
    for p in ("experiments/gpu-sim-u2o.7.1/measured.jsonl",
              "experiments/gpu-sim-e5q/measured.jsonl",
              "experiments/gpu-sim-u2o.11.3/measured.jsonl",
              "experiments/gpu-sim-f6y/measured_new.jsonl"):
        for line in (REPO / p).read_text().splitlines():
            out.add(json.loads(line)["kernel_hash"])
    return out


def main() -> int:
    zv, bwv, ancient, old = _models()
    frozen = frozen_hashes()
    print(f"frozen hashes: {len(frozen)}")
    keep, seen_hash, dropped_collision = [], set(), 0

    def consider(op, n, grid, block, shmem, extra):
        src = SOURCES[op]
        h = kernel_hash_for(src, grid, block, n)
        if h in frozen or h in seen_hash:
            return "collision"
        lc = LaunchConfig(grid_x=grid[0], grid_y=grid[1], grid_z=grid[2],
                          block_x=block[0], block_y=block[1],
                          block_z=block[2], shared_mem_bytes=shmem)
        f = extract_features(src, lc, n)
        cls = classify(f)
        pred_ms = old.roofline_predict(zv.to_old(old, f)).predicted_time_ms
        cell = {"op": op, "n": n, "grid": grid, "block": block,
                "shared_mem_bytes": shmem,
                "predicted_class": cls, "predicted_ms": pred_ms,
                "predicted_measurable": pred_ms >= MEAS_MARGIN_MS,
                "bytes_moved": f.estimated_bytes_moved,
                "kernel_hash": h}
        cell.update(extra)
        keep.append(cell)
        seen_hash.add(h)
        return cls

    counts = Counter()
    for op, extra in HIREF:
        for n in SIZES_HIREF:
            for bb in GRIDS_HIREF:
                gg = n // bb
                shmem = SHMEM_1D.get(op, lambda bb: 0)(bb)
                r = consider(op, n, [gg, 1, 1], [bb, 1, 1], shmem, extra)
                counts[r] += 1
    for op, extra in STREAM:
        for n in SIZES_STREAM:
            for bb in (256, 1024):
                gg = n // bb
                r = consider(op, n, [gg, 1, 1], [bb, 1, 1], 0, extra)
                counts[r] += 1
    for op, extra in ATOMIC_DIV:
        for n in SIZES_ATOMIC:
            for bb in (256, 1024):
                gg = n // bb
                r = consider(op, n, [gg, 1, 1], [bb, 1, 1], 0, extra)
                counts[r] += 1
    for op in ELEM1_OPS:
        for n in SIZES_ELEM1:
            for bb in (256, 1024):
                gg = n // bb
                r = consider(op, n, [gg, 1, 1], [bb, 1, 1], 0, {})
                counts[r] += 1
    for (op, (rows, cols), (bx, by), shmem) in CELLS_2D:
        n = rows * cols
        gx, gy = cols // bx, rows // by
        assert gx * bx == cols and gy * by == rows
        r = consider(op, n, [gx, gy, 1], [bx, by, 1], shmem,
                     {"rows": rows, "cols": cols})
        counts[r] += 1
    # GEMM: custom large cells on the generator's (16,16)-block scheme:
    # grid (M/16, N/16), tile 16, shmem 2048 (known-good, not invented).
    # M/N multiples of 16 keep the tiling exact; K is a loop bound.
    for (op, M, N, K) in [
            ("gemm_naive", 2048, 2048, 512),
            ("gemm_naive", 2048, 2048, 1024),
            ("gemm_naive", 4096, 4096, 256),
            ("gemm_naive", 4096, 1024, 1024),
            ("gemm_naive", 1024, 4096, 1024),
            ("gemm_tiled", 2048, 2048, 512),
            ("gemm_tiled", 2048, 2048, 1024),
            ("gemm_tiled", 4096, 4096, 256),
            ("gemm_tiled", 4096, 1024, 1024),
            ("gemm_tiled", 1024, 4096, 1024)]:
        assert M % 16 == 0 and N % 16 == 0, (M, N)
        r = consider(op, M * N * K, [M // 16, N // 16, 1], [16, 16, 1],
                     2048, {"M": M, "N": N, "K": K, "tile": 16})
        counts[r] += 1
    # GEMM: generator-enumerated large cells (own grids/dims), as generated.
    for s in g4(max_kernels=100000):
        d = s.to_dict()
        dims = {k: v for k, v in (d.get("metadata") or {}).items()
                if type(v) is int}
        if not dims or max(dims.get("M", 0), dims.get("N", 0),
                           dims.get("K", 0)) < 512:
            counts["gemm-skipped-small"] += 1
            continue
        n = d["problem_size"]
        r = consider(d["op_type"], n, d["grid"], d["block"],
                     d.get("shared_mem_bytes", 0), dims)
        counts[r] += 1

    n_mm = sum(1 for k in keep if k["predicted_class"] == "memory->memory")
    n_meas = sum(1 for k in keep if k["predicted_measurable"])
    print(f"candidates kept: {len(keep)} "
          f"(mm={n_mm}, meas@{MEAS_MARGIN_MS}ms={n_meas})")
    print("by family:", json.dumps(dict(Counter(k["op"] for k in keep))))
    (HERE / "keep_list.json").write_text(json.dumps(keep, indent=2))
    gate = {"gate_mm": GATE_MM, "gate_meas": GATE_MEAS,
            "predicted_mm": n_mm, "predicted_measurable": n_meas,
            "pass": n_mm >= GATE_MM and n_meas >= GATE_MEAS}
    (HERE / "prediction.json").write_text(json.dumps(gate, indent=2))
    print("GATE:", json.dumps(gate))
    if not gate["pass"]:
        print("KEEP-LIST SHORT: gate fired -- extend candidates, do not generate")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
