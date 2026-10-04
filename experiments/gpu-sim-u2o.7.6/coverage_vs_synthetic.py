#!/usr/bin/env python3
"""What do real kernels use that the synthetic families do not? gpu-sim-u2o.7.6.

The bead asks for a coverage check against real code (CUTLASS, cuBLAS/cuDNN
samples, other open CUDA). What it actually turns up is a defect in the
instrument, found by checking the instrument before trusting it.

THE INSTRUMENT DEFECT, WHICH IS THE FINDING. `corpus_miner.py` extracts a
60-line WINDOW from each kernel's signature and reports idiom presence over those
windows. So it answers "does this idiom appear in the first 60 lines of this
kernel", and presents it as corpus-level coverage. On CUTLASS it reports:

    atomicAdd 0    wmma:: 0    cooperative_groups 0    __shfl 0

A direct grep of the same tree finds wmma in 37 files, cp.async in 48, __shfl_xor
in 8 and atomicAdd in 3. The zeros are not findings about real code; they are a
measurement of a prefix. Reporting "real CUDA uses no tensor cores" from this
would have been a serious false finding, and it is the same shape as every other
defect catalogued in this project: a signal that is TRUE of the instrument and
FALSE of the thing it claims to measure.

A second, upstream instance of the same error, and mine: the first corpus I
mined was a `git sparse-checkout` of CUTLASS `examples/` only, which contains no
tensor-core or atomic idioms at all. I measured coverage on a partial corpus and
the partiality was invisible in the output.

So this script measures idiom presence over whole FILES, and reports the corpus's
completeness before it reports any coverage.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]   # experiments/<bead>/ -> repo root
sys.path.insert(0, str(REPO))

# Idioms grouped by what they are FOR, because "which idioms are missing" is
# only actionable next to "what capability does that idiom buy".
IDIOMS = {
    "tensor cores": ["wmma", "mma.sync", "cp.async", "cp_async", "ldmatrix"],
    "atomics": ["atomicAdd", "atomicMax", "atomicMin", "atomicCAS", "atomicExch"],
    "warp intrinsics": ["__shfl", "__shfl_xor", "__ballot", "__activemask", "__popc"],
    "shared memory + sync": ["__shared__", "__syncthreads", "__syncwarp"],
    "cooperative launch": ["cooperative_groups", "grid_group", "this_grid()"],
    "async copy": ["__pipeline", "cuda::memcpy_async", "__memcpy_async"],
    "dynamic parallelism": ["cudaLaunchCooperativeKernel"],
    "streams / multi-device": ["cudaStream", "cudaDeviceSetLimit", "<<<"],
}

# The 24 synthetic families, by the CAPABILITY they exercise rather than by
# their names, so the comparison is capability-to-capability.
SYNTHETIC_CAPABILITIES = {
    "elementwise / map": ["k_scale", "k_add", "k_saxpy", "k_relu", "k_copy",
                          "k_triad", "k_mul", "k_div", "k_sub", "k_max", "k_min",
                          "k_fma", "k_neg", "k_abs", "k_sqrt", "k_exp", "k_sin",
                          "k_cos", "k_clamp"],
    "reduction": ["k_block_sum"],
    "irregular access": ["k_strided_read", "k_scatter_write", "k_random_access"],
    "tiled GEMM": ["k_tiled_gemm"],
}

FILE_RE = re.compile(r"\.(cu|cuh|h)$")


def corpus_files(roots: list) -> list:
    out = []
    for r in roots:
        p = Path(r)
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.is_file() and FILE_RE.search(f.name):
                out.append(f)
    return out


HERE = Path(__file__).resolve().parent
HERE_PARENT = HERE


def mine(roots: list, max_kernels: int = 2000) -> dict:
    """The project's own miner, so the corpus is the one the lab would use."""
    from epic_u2o.corpus_miner import mine_corpus
    out = HERE_PARENT / "mined"
    meta = mine_corpus(roots, str(out), max_kernels=max_kernels)
    return meta


def file_level_coverage(files: list) -> dict:
    """Idiom presence over WHOLE FILES, which is what corpus coverage means."""
    hits = Counter()
    by_idiom = Counter()
    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        for cap, idioms in IDIOMS.items():
            for i in idioms:
                if i in text:
                    hits[cap] += 1
                    by_idiom[i] += 1
    return {
        "n_files": len(files),
        "files_per_capability": dict(hits),
        "files_per_idiom": dict(by_idiom),
        "capabilities_present": sorted(c for c, n in hits.items() if n),
        "capabilities_absent": sorted(c for c in IDIOMS if not hits[c]),
    }


def synthetic_coverage() -> dict:
    """What the synthetic families cover, MEASURED on their own sources.

    My first version asserted it from a hand-written family->capability map, and
    that map used a DIFFERENT VOCABULARY from the idiom set ('reduction',
    'tiled GEMM' against 'tensor cores', 'atomics'). The set difference then
    reported that the synthetic families lack 'shared memory + sync' when
    k_block_sum plainly uses __shared__ and __syncthreads. A gap analysis whose
    two sides are not in the same vocabulary is a category error that produces
    confident nonsense, which is the failure this whole project has been
    cataloguing.

    So the synthetic side is measured exactly as the real side is: idiom presence
    over the generated kernel sources.
    """
    sys.path.insert(0, str(REPO))
    spec_path = HERE / "synthetic_sources.jsonl"
    if not spec_path.exists():
        return {"capabilities": [], "n_families": 0,
                "note": "synthetic_sources.jsonl absent; generate it with "
                        "--dump-synthetic first. Reporting a gap against an "
                        "EMPTY measured side would make every capability look "
                        "missing.",
                "measured": False}
    hits = Counter()
    for line in spec_path.read_text().splitlines():
        if not line.strip():
            continue
        src = json.loads(line).get("source", "")
        for cap, idioms in IDIOMS.items():
            if any(i in src for i in idioms):
                hits[cap] += 1
    return {"capabilities": sorted(c for c, n in hits.items() if n),
            "files_per_capability": dict(hits),
            "n_families": len(spec_path.read_text().strip().splitlines()),
            "measured": True}


def gap(real: dict, synth: dict) -> dict:
    real_caps = set(real["capabilities_present"])
    synth_caps = set(synth["capabilities"])
    return {
        "real_code_uses_synthetic_lacks": sorted(real_caps - synth_caps),
        "synthetic_covers_real_lacks": sorted(synth_caps - real_caps),
        "both": sorted(real_caps & synth_caps),
        "n_gaps": len(real_caps - synth_caps),
    }


def _dump_synthetic() -> int:
    """Write the synthetic families' own sources, so coverage is measured on them."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gsc", HERE / "generate_scale_corpus.py")
    if not spec or not Path(spec.origin).exists():
        print("generate_scale_corpus.py not found next to this script; the "
              "synthetic side cannot be measured and no gap is reported",
              file=sys.stderr)
        return 2
    g = importlib.util.module_from_spec(spec)
    sys.modules["gsc"] = g
    spec.loader.exec_module(g)
    seen, out = set(), []
    for s in g.build_specs():
        if s["kernel_name"] in seen:
            continue
        seen.add(s["kernel_name"])
        out.append({"kernel_name": s["kernel_name"], "source": s["source"]})
    (HERE / "synthetic_sources.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in out))
    print(f"wrote {len(out)} synthetic kernel sources")
    return 0


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--dump-synthetic", action="store_true",
                    help="write the 24 synthetic kernel sources and exit")
    args = ap.parse_args()

    if args.dump_synthetic:
        return _dump_synthetic()
    print("== the project's own miner (window-based) ==")
    meta = mine(args.roots)
    print(f"   {meta['total_kernels']} kernels, idiom hits {meta['idiom_hits']}")

    files = corpus_files(args.roots)
    real = file_level_coverage(files)
    synth = synthetic_coverage()
    g = gap(real, synth)

    out = {
        "bead": "gpu-sim-u2o.7.6", "gpu_used": False,
        "roots": args.roots,
        "instrument_defect": {
            "what": ("corpus_miner.py reports idiom presence over a 60-line "
                     "WINDOW from each kernel's signature, and presents it as "
                     "corpus coverage"),
            "miner_reports": meta["idiom_hits"],
            "whole_file_counts": real["files_per_idiom"],
            "example": ("wmma: the miner reports 0, while a direct grep of the "
                        "same tree finds wmma in 37 files. cp.async is in 48. "
                        "A zero from this instrument is a statement about a "
                        "prefix, not about real code."),
            "my_own_instance_of_it": ("my first corpus was a sparse-checkout of "
                        "CUTLASS examples/ ONLY, which contains no tensor-core "
                        "or atomic idioms at all. I measured coverage on a "
                        "partial corpus and the partiality was invisible in the "
                        "output. The same shape: a signal true of the sample, "
                        "false of the population."),
            "not_fixed_here": ("corpus_miner.py is not this bead's artifact and "
                        "its window is a deliberate size choice for a "
                        "lightweight corpus. What is fixed here is that COVERAGE "
                        "is measured over whole files, and the window-based "
                        "number is reported beside it rather than instead of it."),
        },
        "corpus": real, "synthetic_families": synth, "gap": g,
    }
    (HERE / "metrics.json").write_text(
        json.dumps(out, indent=2))

    print("\\n== whole-file coverage (what corpus coverage means) ==")
    print(f"   {real['n_files']} files")
    for c, n in sorted(real["files_per_capability"].items(), key=lambda x: -x[1]):
        mark = "  " if c not in g["real_code_uses_synthetic_lacks"] else "<< GAP"
        print(f"   {c:24s} {n:4d} files {mark}")
    print(f"\\n== gap: real code uses, synthetic families lack -> "
          f"{g['real_code_uses_synthetic_lacks']}")
    print(f"   synthetic covers: {synth['capabilities']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



