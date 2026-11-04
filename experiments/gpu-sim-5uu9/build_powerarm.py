#!/usr/bin/env python3
"""BUILD phase for 5uu9: 12 explicit divergent_branch configurations."""
import argparse, json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from epic_u2o.safe_harness import SafeHarness
from epic_u2o.kernel_driver import synthesize

SOURCE = ("__global__ void divergent_branch(const float* __restrict__ in, "
          "float* __restrict__ out, int n) {\n"
          "    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
          "    if (i < n) {\n"
          "        float v = in[i];\n"
          "        if ((i & 1) == 0) out[i] = v * 2.0f + 1.0f;\n"
          "        else out[i] = v * 0.5f - 1.0f;\n"
          "    }\n"
          "}\n")

CONFIGS = []
for n in (8388608, 16777216, 33554432, 67108864):
    for block in (128, 256, 512):
        grid = (n + block - 1) // block
        CONFIGS.append((n, block, grid))

assert len(CONFIGS) == 12, len(CONFIGS)
sigs = set((n, b, g) for (n, b, g) in CONFIGS)
assert len(sigs) == 12, "duplicate configuration"

def empty_src(g, b):
    return ("#include <cuda_runtime.h>\n#include <stdio.h>\n"
            "__global__ void k_empty() {}\n"
            "int main(){\n"
            "  dim3 grid(%d,%d,%d), block(%d,%d,%d);\n"
            "  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);\n"
            "  for(int w=0;w<20;w++){ k_empty<<<grid,block>>>(); } cudaDeviceSynchronize();\n"
            "  for(int t=0;t<20;t++){\n"
            "    float ms=0; cudaEventRecord(s); k_empty<<<grid,block>>>(); cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e);\n"
            '    printf("{\\"trial\\": %%d, \\"time_ms\\": %%.6f}\\n", t, ms);\n'
            "  }\n  return 0;\n}\n") % (g[0], g[1], g[2], b[0], b[1], b[2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default=None)
    args = ap.parse_args()
    build = Path(args.build_dir) if args.build_dir else (HERE / "build")
    build.mkdir(parents=True, exist_ok=True)
    (build / "bin").mkdir(exist_ok=True)
    h = SafeHarness(timeout_sec=300)
    specs = []
    for idx, (n, block, grid) in enumerate(CONFIGS):
        d = {
            "family": "divergent_branch",
            "kernel_name": "divergent_branch",
            "source": SOURCE,
            "grid": [grid, 1, 1],
            "block": [block, 1, 1],
            "shared_mem_bytes": 0,
            "args": [{"name": "in", "type": "const float*", "ptr": True},
                     {"name": "out", "type": "float*", "ptr": True},
                     {"name": "n", "type": "int", "ptr": False}],
            "problem_size": n,
            "op_type": "divergent_branch",
            "metadata": {"compute_bound": False},
        }
        try:
            prog = synthesize({k: d[k] for k in ("source", "kernel_name", "grid", "block", "args", "problem_size", "shared_mem_bytes", "op_type", "metadata")})
        except Exception as e:
            print(f"synth fail {idx}: {e}", flush=True)
            continue
        exe = build / "bin" / f"k_divergent_branch_{idx:04d}.out"
        ok, msg = h.compile_kernel(prog, str(exe), arch="sm_120")
        if not ok:
            print(f"compile fail {idx}: {msg[:120]}", flush=True)
            continue
        d["exe"] = str(exe)
        d["build_idx"] = idx
        flops = n
        bm = 2 * n * 4.0
        ai = flops / bm if bm > 0 else 0.0
        grid_dim_total = grid
        waves = grid_dim_total / 170.0
        d["phys"] = {"n": n, "block": block, "bm": bm, "flops": flops, "ai": ai, "grid": grid_dim_total, "waves": waves}
        specs.append(d)
        print(f"built {idx}: n={n} block={block} grid={grid}", flush=True)
    # empty baselines
    uniq = set((tuple(d["grid"]), tuple(d["block"])) for d in specs)
    for (g, b) in uniq:
        key = "empty_" + "_".join(map(str, list(g) + list(b)))
        exe = build / "bin" / (key + ".out")
        if exe.exists():
            continue
        ok, msg = h.compile_kernel(empty_src(g, b), str(exe), arch="sm_120")
        if not ok:
            print(f"empty compile fail {key}: {msg[:120]}", flush=True)
    manifest = {"specs": specs}
    (build / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2))
    print(f"DONE: {len(specs)} specs -> {build/'manifest.json'}")

if __name__ == "__main__":
    main()
