#!/usr/bin/env python3
"""u2o.11.3 regime extension builder (CPU): append EXACTLY the keep-list.

Reads mm_keeplist.json (111 predicted memory->memory cells) and appends
one manifest spec per cell to build/manifest.json. The first 2211
occurrences are asserted byte-identical afterwards -- the extension is
purely additive; nothing regenerates, nothing reorders.

Each spec carries the same fields as build_batch.py's (covers via the
identical min(threads, n) rule, source_path via the identical
build/k_<hash>.cu preflight write, deliberately_poor via the identical
[1,1,1] rule) and is synthesized (not just hashed) here: a template/args
mismatch fails now on CPU, not in the GPU slot. Metadata mirrors the
stage generators' own schemas key-for-key so downstream code (driver dims
rule, features, census) treats extension rows identically.

Shared memory equals the predicted values (mm_keeplist records them):
predicting with one shmem and measuring with another would move the class
after the fact.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.generator_stage1 import KernelSpec, kernel_hash_for  # noqa: E402
from epic_u2o.generator_stage2 import generate_kernels as g2  # noqa: E402
from epic_u2o.generator_stage3 import generate_kernels as g3  # noqa: E402
from epic_u2o.generator_stage5 import generate_kernels as g5  # noqa: E402
from epic_u2o.kernel_driver import synthesize  # noqa: E402
from epic_u2o.kernel_names import resolve_kernel_name  # noqa: E402

SOURCES = {}
for gen in (g2, g3, g5):
    for s in gen(max_kernels=100000):
        SOURCES.setdefault(s.op_type, s.to_dict()["source"])

STAGE = {"reduce_max_block": "stage2", "reduce_min_block": "stage2",
         "reduce_sum_block": "stage2", "reduce_sum_warp_shuffle": "stage2",
         "shared_mem_bank_conflict_1way": "stage2",
         "shared_mem_bank_conflict_2way": "stage2",
         "shared_mem_bank_conflict_32way": "stage2",
         "shared_mem_reduction_tiled": "stage2",
         "stencil_2d_5pt": "stage2", "shared_mem_transpose": "stage2",
         "gather_read": "stage3", "random_access": "stage3",
         "scatter_write": "stage3",
         "histogram_global": "stage5", "histogram_shared": "stage5"}

REDUCE_OPS = ("reduce_max_block", "reduce_min_block", "reduce_sum_block",
              "shared_mem_reduction_tiled", "reduce_sum_warp_shuffle")


def args_for(op):
    if op in REDUCE_OPS:
        return [{"name": "in", "type": "const float*", "ptr": True},
                {"name": "out", "type": "float*", "ptr": True},
                {"name": "n", "type": "int", "ptr": False}]
    if op.startswith("shared_mem_bank_conflict"):
        return [{"name": "out", "type": "float*", "ptr": True},
                {"name": "n", "type": "int", "ptr": False}]
    if op in ("stencil_2d_5pt", "shared_mem_transpose"):
        return [{"name": "in", "type": "const float*", "ptr": True},
                {"name": "out", "type": "float*", "ptr": True},
                {"name": "rows", "type": "int", "ptr": False},
                {"name": "cols", "type": "int", "ptr": False}]
    if op in ("gather_read", "scatter_write"):
        return [{"name": "in", "type": "const float*", "ptr": True},
                {"name": "out", "type": "float*", "ptr": True},
                {"name": "indices", "type": "const int*", "ptr": True},
                {"name": "n", "type": "int", "ptr": False}]
    if op == "random_access":
        return [{"name": "in", "type": "const float*", "ptr": True},
                {"name": "out", "type": "float*", "ptr": True},
                {"name": "seeds", "type": "const unsigned int*", "ptr": True},
                {"name": "n", "type": "int", "ptr": False}]
    if op in ("histogram_global", "histogram_shared"):
        return [{"name": "in", "type": "const int*", "ptr": True},
                {"name": "hist", "type": "int*", "ptr": True},
                {"name": "n", "type": "int", "ptr": False},
                {"name": "nbins", "type": "int", "ptr": False}]
    raise ValueError(f"no args rule for {op}")


def metadata_for(op, cell):
    if op in REDUCE_OPS:
        return {"generator": "stage2_reduction_shared", "op_type": op,
                "uses_shared_memory": True,
                "uses_warp_shuffle": "warp_shuffle" in op,
                "bank_conflict_pattern": None, "arithmetic_intensity": 0.5}
    if op.startswith("shared_mem_bank_conflict"):
        pat = op.rsplit("_", 1)[-1]
        assert pat in ("1way", "2way", "32way"), pat
        return {"generator": "stage2_reduction_shared", "op_type": op,
                "uses_shared_memory": True, "uses_warp_shuffle": False,
                "bank_conflict_pattern": pat, "arithmetic_intensity": 0.5}
    if op == "stencil_2d_5pt":
        return {"generator": "stage2_reduction_shared", "op_type": op,
                "uses_shared_memory": True, "stencil_pattern": "5pt",
                "rows": cell["rows"], "cols": cell["cols"]}
    if op == "shared_mem_transpose":
        return {"generator": "stage2_reduction_shared", "op_type": op,
                "uses_shared_memory": True, "access_pattern": "transpose",
                "rows": cell["rows"], "cols": cell["cols"]}
    if op == "gather_read":
        return {"generator": "stage3_memory_access", "access_pattern": "gather",
                "coalesced": False, "indirect": True}
    if op == "random_access":
        return {"generator": "stage3_memory_access", "access_pattern": "random",
                "coalesced": False, "pseudo_random": True}
    if op == "scatter_write":
        return {"generator": "stage3_memory_access", "access_pattern": "scatter",
                "coalesced": False, "indirect": True}
    if op in ("histogram_global", "histogram_shared"):
        return {"generator": "stage5_divergence_atomics", "op_type": op,
                "divergent": False, "uses_atomics": True,
                "uses_shared_memory": op == "histogram_shared",
                "nbins": cell.get("nbins", 0)}
    raise ValueError(f"no metadata rule for {op}")


def main() -> int:
    keep = json.loads((HERE / "mm_keeplist.json").read_text())
    build = HERE / "build"
    old = json.loads((build / "manifest.json").read_text())
    # Rebuild from the COMMITTED base (build/manifest.json may already
    # carry a failed extension attempt): base = non-extension specs.
    base = [d for d in old["specs"] if not d.get("extension")]
    assert len(base) == old.get("n_base", len(base)), \
        "committed base size moved; refusing to stack extensions"
    old_hashes = {d["kernel_hash"] for d in base}
    new_specs, seen = [], set()
    for cell in keep:
        op = cell["op"]
        assert op in STAGE, op
        src = SOURCES[op]
        spec = KernelSpec(
            kernel_name=resolve_kernel_name({"source": src,
                                             "kernel_name": op}),
            source=src, grid=cell["grid"], block=cell["block"],
            shared_mem_bytes=cell.get("shared_mem_bytes", 0),
            args=args_for(op), problem_size=cell["n"], op_type=op,
            metadata=metadata_for(op, cell))
        d = spec.to_dict()
        d["kernel_hash"] = kernel_hash_for(src, d["grid"], d["block"],
                                           cell["n"])
        d["stage"] = STAGE[op]
        # identical rules to build_batch.py:
        threads = (d["grid"][0] * d["grid"][1] * d["grid"][2]
                   * d["block"][0] * d["block"][1] * d["block"][2])
        d["covers"] = min(threads, cell["n"])
        d["deliberately_poor"] = (d["grid"] == [1, 1, 1]
                                  or d["block"] == [1, 1, 1])
        d["extension"] = "mm-regime"
        # synthesize NOW: template/arg mismatches fail on CPU, not in queue.
        prog = synthesize({**d, "metadata": d["metadata"]})
        assert "cudaEventRecord" in prog
        cu = build / f"k_{d['kernel_hash']}.cu"
        cu.write_text(src)
        d["source_path"] = cu.name
        assert d["kernel_hash"] not in old_hashes, \
            f"extension cell collides with existing {d['kernel_hash']}"
        assert d["kernel_hash"] not in seen, \
            f"extension keep-list has duplicate {d['kernel_hash']}"
        seen.add(d["kernel_hash"])
        new_specs.append(d)
    specs = base + new_specs
    assert [d["kernel_hash"] for d in specs[:len(base)]] == \
           [d["kernel_hash"] for d in base], \
        "base occurrences moved: extension must be purely additive"
    out = {"bead": "gpu-sim-u2o.11.3",
           "n_specs": len(specs),
           "n_base": len(base),
           "n_extension": len(new_specs),
           "specs": specs}
    (build / "manifest.json").write_text(json.dumps(out, indent=2))
    print(f"extension: +{len(new_specs)} specs "
          f"({len(base)} -> {len(specs)}), base order preserved, "
          f"all synthesize clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
