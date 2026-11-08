#!/usr/bin/env python3
"""BUILD for 13th config: grid-stride variant at 16M, block 256 (mechanism probe)."""
import argparse, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from epic_u2o.safe_harness import SafeHarness

SOURCE = ("__global__ void divergent_branch(const float* __restrict__ in, "
          "float* __restrict__ out, int n) {\n"
          "    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {\n"
          "        float v = in[i];\n"
          "        if ((i & 1) == 0) out[i] = v * 2.0f + 1.0f;\n"
          "        else out[i] = v * 0.5f - 1.0f;\n"
          "    }\n"
          "}\n")

N, BLOCK, GRID = 16777216, 256, 256

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
    build = Path(args.build_dir) if args.build_dir else (HERE / "build_13th")
    build.mkdir(parents=True, exist_ok=True)
    (build / "bin").mkdir(exist_ok=True)
    h = SafeHarness(timeout_sec=300)
    d = {
        "family": "divergent_branch",
        "kernel_name": "divergent_branch",
        "source": SOURCE,
        "grid": [GRID, 1, 1],
        "block": [BLOCK, 1, 1],
        "shared_mem_bytes": 0,
        "args": [{"name": "in", "type": "const float*", "ptr": True},
                 {"name": "out", "type": "float*", "ptr": True},
                 {"name": "n", "type": "int", "ptr": False}],
        "problem_size": N,
        "op_type": "divergent_branch",
        "metadata": {"compute_bound": False, "mechanism_probe": True,
                     "grid_stride": True},
        "phys": {"n": N, "block": BLOCK, "grid": GRID},
    }
    from epic_u2o.kernel_driver import synthesize
    prog = synthesize({k: d[k] for k in ("source", "kernel_name", "grid", "block", "args", "problem_size", "shared_mem_bytes", "op_type", "metadata")})
    exe = build / "bin" / "k_divergent_branch_13th.out"
    ok, msg = h.compile_kernel(prog, str(exe), arch="sm_120")
    if not ok:
        print(f"compile fail: {msg[:200]}", flush=True); sys.exit(1)
    d["exe"] = str(exe)
    d["build_idx"] = 13
    # empty baseline
    g = (GRID, 1, 1); b = (BLOCK, 1, 1)
    key = "empty_" + "_".join(map(str, list(g) + list(b)))
    empty_exe = build / "bin" / (key + ".out")
    if not empty_exe.exists():
        ok, msg = h.compile_kernel(empty_src(g, b), str(empty_exe), arch="sm_120")
        if not ok:
            print(f"empty compile fail: {msg[:200]}", flush=True); sys.exit(1)
    manifest = {"specs": [d]}
    (build / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2))
    print(f"DONE: 1 spec -> {build/'manifest.json'}")

if __name__ == "__main__":
    main()
