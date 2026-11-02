#!/usr/bin/env python3
"""BUILD phase (CPU, unqueued): generate specs for chosen unseen families,
synthesize drivers, compile binaries, write manifest. No GPU contact."""
import argparse, json, sys, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from epic_u2o.kernel_driver import synthesize
from epic_u2o.safe_harness import SafeHarness

GENS = {
    "strided_read": ("epic_u2o.generator_stage3", "strided_read"),
    "interleaved_read": ("epic_u2o.generator_stage3", "interleaved_read"),
    "block_cyclic": ("epic_u2o.generator_stage3", "block_cyclic"),
    "gemm_naive": ("epic_u2o.generator_stage4", "gemm_naive"),
    "divergent_branch": ("epic_u2o.generator_stage5", "divergent_branch"),
}


def gen_specs(family, n_target, min_size=0):
    modname, op = GENS[family]
    mod = __import__(modname, fromlist=["generate_kernels"])
    out = []
    for spec in mod.generate_kernels(max_kernels=100000):
        if spec.op_type != op:
            continue
        if min_size and spec.problem_size < min_size:
            continue
        out.append((family, spec.to_dict()))
        if len(out) >= n_target:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="strided_read")
    ap.add_argument("--n-per-family", type=int, default=120)
    ap.add_argument("--min-size", type=int, default=0)
    ap.add_argument("--build-dir", default=None)
    args = ap.parse_args()
    build = Path(args.build_dir) if args.build_dir else (HERE / "build")
    build.mkdir(parents=True, exist_ok=True)
    (build / "bin").mkdir(exist_ok=True)
    h = SafeHarness(timeout_sec=300)
    specs_all = []
    for fam in args.families.split(","):
        fam = fam.strip()
        rows = gen_specs(fam, args.n_per_family, args.min_size)
        print(f"{fam}: generated {len(rows)} specs", flush=True)
        for i, (fam, d) in enumerate(rows):
            d = dict(d)
            d["family"] = fam
            try:
                prog = synthesize({k: d[k] for k in ("source", "kernel_name", "grid", "block", "args", "problem_size", "shared_mem_bytes", "op_type", "metadata")})
            except Exception as e:
                print(f"  synth fail {fam} {i}: {e}", flush=True)
                continue
            exe = build / "bin" / f"k_{fam}_{i:04d}.out"
            c, m = h.compile_kernel(prog, str(exe), arch="sm_120")
            if not c:
                print(f"  compile fail {fam} {i}: {m[:120]}", flush=True)
                continue
            d["exe"] = str(exe)
            d["build_idx"] = i
            n = d["problem_size"]
            grid = d["grid"]; block = d["block"]
            if fam == "gemm_naive":
                md = d.get("metadata", {})
                M = md.get("M", 1); N = md.get("N", 1); K = md.get("K", 1)
                flops = M * N * K
            else:
                flops = n
            bm = 2 * n * 4.0
            ai = flops / bm if bm > 0 else 0.0
            grid_dim_total = grid[0] * grid[1] * grid[2]
            threads = grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2]
            waves = grid_dim_total / 170.0
            d["phys"] = {"n": n, "block": block[0] * block[1] * block[2],
                         "bm": bm, "flops": flops, "ai": ai, "grid": grid_dim_total, "waves": waves}
            specs_all.append(d)
            if (i + 1) % 50 == 0:
                print(f"  {fam}: compiled {i + 1}", flush=True)
    manifest = {"specs": specs_all}
    (build / "manifest.json").write_text(json.dumps(manifest))
    print(f"total compiled specs: {len(specs_all)} -> {build / 'manifest.json'}")
    # empty baselines (20 trials, 1 launch each, matching kernel_driver)
    uniq = set()
    for d in specs_all:
        uniq.add((tuple(d["grid"]), tuple(d["block"])))
    print(f"unique launch configs: {len(uniq)}")
    for (g, b) in uniq:
        key = "empty_" + "_".join(map(str, list(g) + list(b)))
        cu = build / "bin" / (key + ".cu")
        exe = build / "bin" / (key + ".out")
        if exe.exists():
            continue
        src = (
            '#include <cuda_runtime.h>\n'
            '#include <stdio.h>\n'
            '__global__ void k_empty() {}\n'
            'int main(){\n'
            '  dim3 grid(%d,%d,%d), block(%d,%d,%d);\n'
            '  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);\n'
            '  for(int w=0;w<20;w++){ k_empty<<<grid,block>>>(); } cudaDeviceSynchronize();\n'
            '  for(int t=0;t<20;t++){\n'
            '    float ms=0; cudaEventRecord(s); k_empty<<<grid,block>>>(); cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e);\n'
            '    printf("{\\"trial\\": %d, \\"time_ms\\": %.6f}\\n", t, ms);\n'
            '  }\n'
            '  return 0;\n'
            '}\n'
        ) % (g[0], g[1], g[2], b[0], b[1], b[2])
        cu.write_text(src)
        r = subprocess.run(["nvcc", "-arch=sm_120", "-O2", str(cu), "-o", str(exe)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  empty compile fail {g},{b}: {r.stderr[:120]}", flush=True)
    print("empty baselines ready")


if __name__ == "__main__":
    main()
