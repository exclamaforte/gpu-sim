#!/usr/bin/env python3
"""Build divergent_branch specs with SIZE-SCALING grid: grid=ceil(n/256), block=256,
n in 1M-16M. Per-thread work = n/grid = 256 elements, kernel should scale."""
import json, sys, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import os
os.chdir(REPO)
from epic_u2o.kernel_driver import synthesize
from epic_u2o.safe_harness import SafeHarness
import importlib.util
spec = importlib.util.spec_from_file_location("g5", REPO / "epic_u2o" / "generator_stage5.py")
g5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g5)

sizes = [1048576, 2097152, 4194304, 8388608, 16777216]
_build_dir_env = os.environ.get("HSUV_BUILD_DIR")
build = Path(_build_dir_env) if _build_dir_env else (HERE / "build_div_sizescale")
build.mkdir(parents=True, exist_ok=True)
(build / "bin").mkdir(exist_ok=True)
h = SafeHarness(timeout_sec=300)
specs_all = []
idx = 0
for spec_obj in g5.generate_kernels(max_kernels=100000):
    if spec_obj.op_type != "divergent_branch":
        continue
    d = spec_obj.to_dict()
    d["family"] = "divergent_branch"
    n = sizes[idx % len(sizes)]
    d["problem_size"] = n
    block = 256
    grid = (n + block - 1) // block
    d["grid"] = [grid, 1, 1]
    d["block"] = [block, 1, 1]
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
    _sr = subprocess.run(["strip","-s",str(exe)], capture_output=True)
    if _sr.returncode != 0:
        print("strip fail", idx, _sr.stderr.decode(errors="replace")[:120])
    try:
        d["exe"] = str(exe.relative_to(build))
    except ValueError:
        d["exe"] = exe.name
    d["build_idx"] = idx - 1
    bm = 2 * n * 4.0
    flops = n
    ai = flops / bm
    grid_dim_total = grid
    waves = grid_dim_total / 170.0
    d["phys"] = {"n": n, "block": block, "bm": bm, "flops": flops, "ai": ai, "grid": grid_dim_total, "waves": waves}
    specs_all.append(d)
    if len(specs_all) >= 56:
        break
manifest = {"specs": specs_all}
(build / "manifest.json").write_text(json.dumps(manifest))
print(f"compiled {len(specs_all)} size-scaling specs")
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
    r = subprocess.run(["nvcc","-arch=sm_120","-O2","-s",str(cu),"-o",str(exe)], capture_output=True, text=True)
    if r.returncode != 0:
        print("empty compile fail", g, b, r.stderr[:120])
print("empty baselines ready")
