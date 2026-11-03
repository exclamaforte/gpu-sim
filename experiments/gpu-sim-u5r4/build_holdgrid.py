#!/usr/bin/env python3
"""Build divergent_branch specs with a FIXED grid (64 blocks x 256 threads) across
1M-16M sizes, so per-thread work scales with size (hold-grid arm)."""
import json, sys, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from epic_u2o.kernel_driver import synthesize
from epic_u2o.safe_harness import SafeHarness
import importlib.util
spec = importlib.util.spec_from_file_location("g5", REPO / "epic_u2o" / "generator_stage5.py")
g5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g5)

sizes = [1048576, 2097152, 4194304, 8388608, 16777216]
build = HERE / "build_div_holdgrid"
build.mkdir(parents=True, exist_ok=True)
(build / "bin").mkdir(exist_ok=True)
h = SafeHarness(timeout_sec=300)
specs_all = []
idx = 0
# generate kernels from the original generator but override grid/block
for spec_obj in g5.generate_kernels(max_kernels=100000):
    if spec_obj.op_type != "divergent_branch":
        continue
    d = spec_obj.to_dict()
    d["family"] = "divergent_branch"
    d["grid"] = [64, 1, 1]
    d["block"] = [256, 1, 1]
    # pick sizes in range by cycling
    d["problem_size"] = sizes[idx % len(sizes)]
    idx += 1
    if idx > 56:
        break
    try:
        prog = synthesize({k: d[k] for k in ("source", "kernel_name", "grid", "block", "args", "problem_size", "shared_mem_bytes", "op_type", "metadata")})
    except Exception as e:
        print("synth fail", idx, e); continue
    exe = build / "bin" / f"k_divergent_branch_{idx-1:04d}.out"
    c, m = h.compile_kernel(prog, str(exe), arch="sm_120")
    if not c:
        print("compile fail", idx, m[:120]); continue
    d["exe"] = str(exe)
    d["build_idx"] = idx - 1
    n = d["problem_size"]
    grid = d["grid"]; block = d["block"]
    bm = 2 * n * 4.0
    flops = n
    ai = flops / bm
    grid_dim_total = grid[0]*grid[1]*grid[2]
    waves = grid_dim_total / 170.0
    d["phys"] = {"n": n, "block": block[0]*block[1]*block[2], "bm": bm, "flops": flops, "ai": ai, "grid": grid_dim_total, "waves": waves}
    specs_all.append(d)
    if len(specs_all) >= 56:
        break
manifest = {"specs": specs_all}
(build / "manifest.json").write_text(json.dumps(manifest))
print(f"compiled {len(specs_all)} hold-grid specs")
# empty baselines
uniq = set()
for d in specs_all:
    uniq.add((tuple(d["grid"]), tuple(d["block"])))
for (g, b) in uniq:
    key = "empty_" + "_".join(map(str, list(g)+list(b)))
    cu = build / "bin" / (key + ".cu")
    exe = build / "bin" / (key + ".out")
    if exe.exists(): continue
    src = ('#include <cuda_runtime.h>\n#include <stdio.h>\n__global__ void k_empty() {}\n'
           'int main(){\n  dim3 grid(%d,%d,%d), block(%d,%d,%d);\n'
           '  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);\n'
           '  for(int w=0;w<20;w++){ k_empty<<<grid,block>>>(); } cudaDeviceSynchronize();\n'
           '  for(int t=0;t<20;t++){\n    float ms=0; cudaEventRecord(s); k_empty<<<grid,block>>>(); cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e);\n'
           '    printf("{\\"trial\\": %%d, \\"time_ms\\": %%.6f}\\n", t, ms);\n  }\n  return 0;\n}\n') % (g[0],g[1],g[2],b[0],b[1],b[2])
    cu.write_text(src)
    r = subprocess.run(["nvcc","-arch=sm_120","-O2",str(cu),"-o",str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print("empty compile fail", g, b, r.stderr[:120])
print("empty baselines ready")
