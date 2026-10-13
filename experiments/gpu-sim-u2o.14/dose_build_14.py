#!/usr/bin/env python3
"""gpu-sim-u2o.14 DOSE BUILD (CPU, unqueued): 48 stratified stage-1 specs.

Rule (stated, not cherry-picked): 8 elementwise ops x n {2^12, 2^18} x
block {64, 256, 1024}, grid = full coverage. kernel_hash via the canonical
kernel_hash_for. Compiles drivers with nvcc here; the queued job only
measures prebuilt binaries. Writes dose_manifest.json + build/k_*.out.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.generator_stage1 import generate_kernels, kernel_hash_for  # noqa: E402
from epic_u2o.kernel_driver import synthesize  # noqa: E402

OPS = ["scale", "add", "saxpy", "relu", "copy", "triad", "mul", "div"]
SIZES = [1 << 12, 1 << 18]
BLOCKS = [[64, 1, 1], [256, 1, 1], [1024, 1, 1]]
ARCH = "sm_120"


def main() -> int:
    build = HERE / "build"
    build.mkdir(exist_ok=True)
    specs = []
    for op in OPS:
        for n in SIZES:
            for block in BLOCKS:
                full = max(1, n // (block[0] * block[1] * block[2]))
                for s in generate_kernels(
                        ops=[op], grid_sweeps=[[full, 1, 1]],
                        block_sweeps=[block], problem_sizes=[n]):
                    d = s.to_dict()
                    d["kernel_hash"] = kernel_hash_for(d["source"], d["grid"],
                                           d["block"], d["problem_size"])
                    cu = build / f"k_{d['kernel_hash']}.cu"
                    cu.write_text(synthesize(d))
                    specs.append({k: d[k] for k in
                                  ("kernel_hash", "kernel_name", "op_type",
                                   "grid", "block", "problem_size",
                                   "shared_mem_bytes", "source")})
    assert len(specs) == 48, len(specs)
    fails = []
    for d in specs:
        cu = build / f"k_{d['kernel_hash']}.cu"
        exe = build / f"k_{d['kernel_hash']}.out"
        r = subprocess.run(["nvcc", "-O2", f"-arch={ARCH}", str(cu),
                            "-o", str(exe)],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            fails.append({"hash": d["kernel_hash"], "error": r.stderr[:200]})
    (HERE / "dose_manifest.json").write_text(json.dumps(
        {"bead": "gpu-sim-u2o.14", "rule": "8 ops x n{12,18} x block{64,256,1024}, full-coverage grid",
         "n_specs": len(specs), "n_compiled": len(specs) - len(fails),
         "failures": fails, "specs": specs}, indent=2))
    print(f"dose: {len(specs)} specs, {len(fails)} compile failures")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
