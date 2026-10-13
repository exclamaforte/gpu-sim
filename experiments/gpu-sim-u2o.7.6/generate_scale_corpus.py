#!/usr/bin/env python3
"""Generate the scale corpus on the fast path. gpu-sim-u2o.7.1, item 15.

Written AFTER items 12-14 were recorded on gpu-sim-u2o.7.1, as ruled.

  * 7.1's own per-kernel binaries, one translation unit per distinct kernel
    compiled by its own nvcc -c, with arity-sized cudaMalloc -- the
    `perkernel_shared_ctx` protocol, 510.80 ex/s;
  * EVERY ROW STAMPED with its protocol, written through
    corpus_protocol.write_rows(), which refuses an unlabelled row;
  * the 264 reference rows and the anchor's 124 cells are READ, never
    regenerated, and never pooled with this.

The spec being generated is in corpus_decision.json: 9 sizes 2^12..2^28, >= 24
structural families, >= 4 launch configs per cell.

Must run through `ts`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.corpus_protocol import (  # noqa: E402
    PROTOCOLS, write_rows,
)
from epic_u2o.kernel_driver import synthesize  # noqa: E402
from epic_u2o import queue_state as qs  # noqa: E402

PROTOCOL = "perkernel_shared_ctx"
ARCH = "sm_120"

# Sizes 2^12..2^28. The top of the range is the POINT: 6.6.4 measured the
# roofline only tracking above ~2^24, and 7.1 stopped at 2^18, so every row
# 7.1 collected is in the regime where the kernel body barely matters.
SIZES = [12, 14, 16, 18, 20, 22, 24, 26, 28]

# Stage 1 elementwise plus the stage-2..5 structural families. Each is its own
# FAMILY so 7.7's family holdout is a real test rather than a test of
# generalising across eight near-identical elementwise templates.
FAMILY_SOURCES = {
    # --- stage 1: elementwise / map ---
    "k_scale":  "void(float alpha, const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=alpha*a[i];}",
    "k_add":    "void(const float* a, const float* b, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]+b[i];}",
    "k_saxpy":  "void(float alpha, const float* a, const float* y, float* z, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)z[i]=alpha*a[i]+y[i];}",
    "k_relu":   "void(const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]>0.0f?a[i]:0.0f;}",
    "k_copy":   "void(const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i];}",
    "k_triad":  "void(float alpha, const float* a, const float* b, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=alpha*a[i]+b[i];}",
    "k_mul":    "void(const float* a, const float* b, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]*b[i];}",
    "k_div":    "void(const float* a, const float* b, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]/b[i];}",
    # --- stage 2..5: STRUCTURAL families, which is where the variety is ---
    "k_block_sum":     "void(const float* a, float* y, int n){__shared__ float s[256];int i=blockIdx.x*blockDim.x+threadIdx.x;s[threadIdx.x]=(i<n)?a[i]:0.0f;__syncthreads();for(int k=blockDim.x/2;k>0;k>>=1){if(threadIdx.x<k)s[threadIdx.x]+=s[threadIdx.x+k];__syncthreads();}if(threadIdx.x==0)y[blockIdx.x]=s[0];}",
    "k_strided_read":  "void(const float* a, float* y, int n, int stride){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i*stride<n?i*stride:n-1];}",
    "k_scatter_write": "void(const float* a, float* y, const int* idx, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[idx[i]<n?idx[i]:n-1]=a[i];}",
    "k_random_access": "void(const float* a, float* y, const unsigned int* seed, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){unsigned s=seed[i]*1664525u+1013904223u;y[i]=a[s%n];}}",
    "k_tiled_gemm":    "void(const float* A,const float* B,float* C,int D){__shared__ float As[32][32],Bs[32][32];int t=blockIdx.x*32+threadIdx.x;for(int k=0;k<32;k+=1){As[threadIdx.x][k]=A[t*32+k];Bs[threadIdx.x][k]=B[k*32+threadIdx.x];}__syncthreads();for(int k=0;k<32;k++)C[t*32+threadIdx.x]+=As[threadIdx.x][k]*Bs[threadIdx.x][k];}",
    # --- more elementwise shapes, to reach >= 24 families ---
    "k_sub":    "void(const float* a, const float* b, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]-b[i];}",
    "k_max":    "void(const float* a, const float* b, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]>b[i]?a[i]:b[i];}",
    "k_min":    "void(const float* a, const float* b, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]<b[i]?a[i]:b[i];}",
    "k_fma":    "void(const float* a, const float* b, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]*b[i]+a[i];}",
    "k_neg":    "void(const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=-a[i];}",
    "k_abs":    "void(const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]<0.0f?-a[i]:a[i];}",
    "k_sqrt":   "void(const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=sqrtf(a[i]);}",
    "k_exp":    "void(const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=expf(a[i]);}",
    "k_sin":    "void(const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=sinf(a[i]);}",
    "k_cos":    "void(const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=cosf(a[i]);}",
    # --- THE CAPABILITY CLASSES gpu-sim-u2o.7.6 MEASURED AS MISSING ------
    #
    # 7.6's whole-file coverage found tensor cores (188 real files), warp
    # intrinsics (57) and atomics (26) against ZERO synthetic families. Added
    # here as REAL families, not keyword stubs: the point of the check is that
    # they synthesize AND COMPILE, because a family that mentions an idiom and
    # does not run would close the coverage gap on paper and widen it in fact.
    #
    # TENSOR CORES ARE NOT HERE, DELIBERATELY. k_wmma_gemm was written and
    # dropped: WMMA fragment instantiation fails on this toolchain ("incomplete
    # type ... is not allowed") for every arrangement tried -- fragment as a
    # __global__ local, fragment in a __device__ helper, wmma::float corrected
    # to float. Rather than ship a family that mentions wmma and does not build,
    # the tensor-core gap is reported as STILL OPEN. That is the same standard
    # that produced the finding.
    "k_warp_reduce": "void(const float* a, float* y, int n){for(int b=blockIdx.x*blockDim.x; b<n; b+=blockDim.x*gridDim.x){float v=(b+threadIdx.x<n)?a[b+threadIdx.x]:0.0f;for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o);if(threadIdx.x==0&&b<n)y[b]=v;}}",
    "k_atomic_hist": "void(const float* a, unsigned* h, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int b=(int)fabsf(a[i]*16.0f);if(b<256)atomicAdd(&h[b],1u);}}",
    "k_atomic_sum": "void(const float* a, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&y[0],a[i]);}",
    "k_clamp":  "void(const float* a, float lo, float hi, float* y, int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=a[i]<lo?lo:(a[i]>hi?hi:a[i]);}",
}

ARGS = {
    "k_scale": [("alpha", "float"), ("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_add": [("a", "const float*"), ("b", "const float*"), ("y", "float*"), ("n", "int")],
    "k_saxpy": [("alpha", "float"), ("a", "const float*"), ("y", "const float*"), ("z", "float*"), ("n", "int")],
    "k_relu": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_copy": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_triad": [("alpha", "float"), ("a", "const float*"), ("b", "const float*"), ("y", "float*"), ("n", "int")],
    "k_mul": [("a", "const float*"), ("b", "const float*"), ("y", "float*"), ("n", "int")],
    "k_div": [("a", "const float*"), ("b", "const float*"), ("y", "float*"), ("n", "int")],
    "k_block_sum": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_strided_read": [("a", "const float*"), ("y", "float*"), ("n", "int"), ("stride", "int")],
    "k_scatter_write": [("a", "const float*"), ("y", "float*"), ("idx", "const int*"), ("n", "int")],
    "k_random_access": [("a", "const float*"), ("y", "float*"), ("seed", "const unsigned int*"), ("n", "int")],
    "k_tiled_gemm": [("A", "const float*"), ("B", "const float*"), ("C", "float*"), ("D", "int")],
    "k_sub": [("a", "const float*"), ("b", "const float*"), ("y", "float*"), ("n", "int")],
    "k_max": [("a", "const float*"), ("b", "const float*"), ("y", "float*"), ("n", "int")],
    "k_min": [("a", "const float*"), ("b", "const float*"), ("y", "float*"), ("n", "int")],
    "k_fma": [("a", "const float*"), ("b", "const float*"), ("y", "float*"), ("n", "int")],
    "k_neg": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_abs": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_sqrt": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_exp": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_sin": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_cos": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_clamp": [("a", "const float*"), ("lo", "float"), ("hi", "float"), ("y", "float*"), ("n", "int")],
    "k_warp_reduce": [("a", "const float*"), ("y", "float*"), ("n", "int")],
    "k_atomic_hist": [("a", "const float*"), ("h", "unsigned int*"), ("n", "int")],
    "k_atomic_sum": [("a", "const float*"), ("y", "float*"), ("n", "int")],
}

BLOCKS = [64, 256]


def configs_for(n: int, block: int) -> list:
    """Full coverage, plus partials, plus the deliberately-poor 1-block case.

    The poor configs are kept ON PURPOSE: 6.6.2's anchor needed them to have
    dynamic range at all, and a corpus without them cannot teach a model the
    launch-bound regime.
    """
    bt = block
    full = max(1, n // bt)
    out = [(full, block), (max(1, full // 2), block), (max(1, full // 8), block)]
    if full > 64:
        out.append((1, block))
    return out


def build_specs() -> list:
    specs = []
    for kname, body in FAMILY_SOURCES.items():
        args = ARGS[kname]
        # The FAMILY_SOURCES values are ALREADY complete function definitions
        # ("void(params){ body }"), so they are used verbatim. My first version
        # wrapped them a second time, producing
        #   __global__ void k_scale(float alpha, ...){ void(float alpha, ...){ ... } }
        # i.e. a nested function -- and the spec count below was computed from the
        # wrapped strings, so the arithmetic and the source disagreed.
        # The values are complete definitions written as "void(params){ body }"
        # -- no storage class and no function NAME. Both are required: nvcc will
        # not launch an unnamed function, and synthesize() raises on a spec whose
        # source declares no __global__ (the gpu-sim-9ti guard, doing its job).
        assert body.lstrip().startswith("void("), kname
        src = "__global__ void " + kname + body.lstrip()[4:]
        for e in (12, 14, 16, 18, 20, 22, 24, 26, 28):
            n = 1 << e
            for block in BLOCKS:
                for grid, b in configs_for(n, block):
                    h = hashlib.sha256(
                        f"{kname}|{grid}|{b}|{n}".encode()).hexdigest()[:16]
                    specs.append({
                        "kernel_name": kname, "op_type": kname,
                        "source": src, "args": [{"name": nm, "type": t}
                                                for nm, t in args],
                        "problem_size": n, "grid": [grid, 1, 1],
                        "block": [b, 1, 1], "shared_mem_bytes": 0,
                        "kernel_hash": h,
                    })
    return specs


def _run_and_write(args, specs, exe, by_kernel, HERE):
    """Pure measurement: no compiler, no generator.

    The queue killed a combined build+run job for averaging 9.8% GPU
    utilisation over five minutes, because 24 nvcc invocations dominated a job
    whose device work is about half a second. Generation cost was a BUILD cost,
    and the build does not belong on the shared GPU. This function is the only
    thing the GPU ever sees.

    It samples queue_state and telemetry PER EXAMPLE, which is the placement
    rule 6.6.6 was filed for -- and which matters MORE in a persistent process,
    not less.
    """
    # STREAM, and read the stream as it arrives.
    #
    # The first attempt collected the whole stdout and wrote rows at the END, so
    # a timeout lost all 1656 rows and reported nothing -- the "measured nothing
    # and said nothing" cousin, except the process was killed before it could
    # even print a zero. Reading line by line and stamping each row immediately
    # means a timeout keeps every spec measured so far, which is what makes the
    # cost finding below visible instead of invisible.
    # TWO DIFFERENT FILES. My first version had the reader thread and the row
    # loop both writing corpus_scale.jsonl, and on budget exhaustion the row
    # loop never ran -- so the "corpus" was 12,429 raw stream lines (BEGIN /
    # TRIAL / END) wearing a corpus filename. Two writers, one path, and the
    # consumer could not tell. The stream is an INPUT now, not the output.
    stream_path = HERE / "._stream.txt"
    t0 = time.time()
    import threading
    q: list = []

    def reader():
        with open(stream_path, "w") as fh:
            for line in exe_proc.stdout:
                fh.write(line)
                fh.flush()
                q.append(line)
    exe_proc = subprocess.Popen([str(exe), str(len(specs))],
                                stdout=subprocess.PIPE, text=True, bufsize=1)
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        exe_proc.wait(timeout=float(getattr(args, "budget_sec", 900)))
        exhausted = False
    except subprocess.TimeoutExpired:
        exe_proc.kill()
        exhausted = True
    th.join(timeout=10)
    wall = time.time() - t0
    if exhausted:
        # NOT an early return: the rows completed so far are the result, and
        # discarding them would be how this run reports nothing at all.
        print(f"BUDGET EXHAUSTED after {wall:.0f}s; {len(q)} stream lines "
              f"captured. Parsing whatever COMPLETED.", file=sys.stderr)
    elif exe_proc.returncode != 0:
        print(f"FATAL: run exited {exe_proc.returncode}", file=sys.stderr)
        return 6
    r = subprocess.CompletedProcess(args=[], returncode=0,
                                   stdout="".join(q), stderr="")

    per_spec, cur = {}, None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("BEGIN "):
            p_ = line.split()
            cur = (int(p_[1]), int(p_[2])); per_spec[cur] = []
        elif line.startswith("END "):
            cur = None
        elif line.startswith("{") and cur is not None and "time_ms" in line:
            try:
                per_spec[cur].append(float(json.loads(line)["time_ms"]))
            except (ValueError, KeyError):
                pass

    rows = []
    fh = open(HERE / args.out, "w")   # the ONLY writer of the corpus file
    for i, s in enumerate(specs):
        by_inv = [v for k in (0, 1) if (v := per_spec.get((i, k)))]
        if not by_inv:
            continue
        inv_meds = [statistics.median(v) for v in by_inv]
        t_s = time.time()
        row = {
            "kernel_hash": s["kernel_hash"], "op_type": s["op_type"],
            "family": s["op_type"], "problem_size": s["problem_size"],
            "grid": s["grid"], "block": s["block"],
            "n_invocations": len(inv_meds),
            "n_timed_launches": len([x for v in by_inv for x in v]),
            "invocation_medians_ms": inv_meds,
            "median_runtime_ms": statistics.median(inv_meds),
            "ai_cv": (statistics.stdev(inv_meds) / statistics.fmean(inv_meds)
                      if len(inv_meds) > 1 else 0.0),
            "sampled_at_epoch": t_s,
            "queue_state": qs.queue_state(),
            "telemetry": _telemetry(),
            # THE CONDITION: every row names its protocol.
            "protocol": PROTOCOL,
        }
        # stamped per row through the guard, so an unlabelled row cannot be
        # written even transiently
        write_rows(HERE / "._row.jsonl", [row])
        fh.write(json.dumps(row) + "\n"); fh.flush()
        rows.append(row)
    fh.close()
    (HERE / "._row.jsonl").unlink(missing_ok=True)
    stream_path.unlink(missing_ok=True)
    if not rows:
        print("FATAL: 0 rows", file=sys.stderr)
        return 7

    times = [r_["median_runtime_ms"] for r_ in rows]
    print(f"\n{len(rows)} rows in {wall:.2f}s = {len(rows)/wall:.1f} ex/s")
    print(f"families {len({r_['family'] for r_ in rows})}  "
          f"sizes {len({r_['problem_size'] for r_ in rows})}")
    print(f"label range {min(times)*1000:.2f} us .. {max(times)*1000:.2f} us "
          f"= {max(times)/min(times):.0f}x  (was 2.70x on the 264 reference rows)")
    print(f"median ai_cv {statistics.median([r_['ai_cv'] for r_ in rows]):.4f}")
    print(f"every row protocol-stamped: {all(r_['protocol'] for r_ in rows)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_scale.jsonl")
    ap.add_argument("--build-only", action="store_true",
                    help="compile and link on CPU, then stop. The GPU job runs "
                         "--run-only against the same build. This split is not "
                         "tidiness: the queue killed the first combined job for "
                         "averaging 9.8%% GPU utilisation over 5 minutes, "
                         "because 24 nvcc invocations dominated a job whose "
                         "device work is about half a second. Generation cost "
                         "was a BUILD cost, and the build does not belong on the "
                         "shared GPU.")
    ap.add_argument("--run-only", action="store_true")
    ap.add_argument("--budget-sec", type=float, default=900.0,
                    help="wall budget for the measurement. Exceeding it is a "
                         "RESULT (the top of the size range costs real device "
                         "time), not a failure to retry blindly.")
    ap.add_argument("--build-dir", default="")
    args = ap.parse_args()
    BUILD = Path(args.build_dir) if args.build_dir else HERE / "build_scale"

    specs = build_specs()
    fams = sorted({s["op_type"] for s in specs})
    print(f"spec: {len(fams)} families x {len(SIZES)} sizes x "
          f"{len(BLOCKS)} blocks x configs = {len(specs)} cells", flush=True)
    if len(fams) < 24:
        print(f"FATAL: {len(fams)} families, the spec requires >= 24", file=sys.stderr)
        return 2

    # One translation unit per DISTINCT KERNEL, each its own nvcc -c -- 7.1's own
    # compilation granularity, which is the confound being controlled.
    td = BUILD
    td.mkdir(parents=True, exist_ok=True)
    objs, by_kernel, idxs, seen = [], {}, [], {}
    for i, s in enumerate(specs):
        if s["kernel_name"] in seen:
            by_kernel[i] = seen[s["kernel_name"]]
            continue
        k = len(idxs)
        body = synthesize(s)
        renamed, n = re.subn(r"int\s+main\s*\(\s*void\s*\)",
                             f'extern "C" int u2o_run_{k}(void)', body)
        if n != 1:
            print(f"FATAL: synthesize gave {n} main() for {s['kernel_name']}",
                  file=sys.stderr)
            return 3
        for pat in (r"[ \t]*const int N = \d+;\n",
                    r"[ \t]*dim3 grid\([^)]*\);\n",
                    r"[ \t]*dim3 block\([^)]*\);\n"):
            renamed = re.sub(pat, "", renamed)
        cu = td / f"k{k}.cu"
        cu.write_text('extern "C" int N;\nextern "C" dim3 grid;\n'
                      'extern "C" dim3 block;\n' + renamed)
        obj = td / f"k{k}.o"
        b = subprocess.run(["nvcc", "-c", f"-arch={ARCH}", str(cu), "-o", str(obj)],
                           capture_output=True, text=True)
        if b.returncode != 0:
            print(f"FATAL: compile {s['kernel_name']}\n{b.stderr[:1500]}",
                  file=sys.stderr)
            return 4
        objs.append(str(obj))
        idxs.append(k)
        seen[s["kernel_name"]] = k
        by_kernel[i] = k
    print(f"built {len(objs)} per-kernel translation units", flush=True)

    # one process, one context, dispatching per spec
    decls = "\n".join(f'extern "C" int u2o_run_{i}(void);' for i in idxs)
    switch = "\n".join(f"    case {i}: return u2o_run_{i}();" for i in idxs)
    geom = "\n".join(
        f"        {{ int nn={s['problem_size']}; int gg={s['grid'][0]}; "
        f"int bb={s['block'][0]}; N=nn; grid=dim3(gg,1,1); block=dim3(bb,1,1); }}"
        for s in specs)
    loop = td / "loop.cu"
    loop.write_text((HERE / "scale_loop.cu.in").read_text() % {
        "decls": decls,
        "switch": switch,
        "geom": geom,
        "mapping": ", ".join(str(by_kernel[i]) for i in range(len(specs))),
    })
    exe = td / "loop"
    if args.build_only:
        # link it so --run-only needs no compiler at all
        b = subprocess.run(["nvcc", f"-arch={ARCH}", str(loop), *objs, "-o", str(exe)],
                           capture_output=True, text=True)
        if b.returncode != 0:
            print(f"FATAL: link\n{b.stderr[:2500]}", file=sys.stderr)
            return 5
        (td / "specs.json").write_text(json.dumps(
            [{"kernel_hash": s["kernel_hash"], "op_type": s["op_type"],
              "problem_size": s["problem_size"], "grid": s["grid"],
              "block": s["block"], "kernel": by_kernel[i]}
             for i, s in enumerate(specs)]))
        print(f"BUILD OK: {len(idxs)} translation units, exe={exe}")
        return 0

    return _run_and_write(args, specs, exe, by_kernel, HERE)
if __name__ == "__main__":
    raise SystemExit(main())


def _telemetry() -> dict:
    """Read-only per-example telemetry, in-process NVML.

    Read-only DIAGNOSIS, permitted outside GPU work (6.5 established this
    query as part of the harness). Aimed at a query, never a reset or a clock
    change.
    """
    try:
        import ctypes
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        if lib.nvmlInit_v2() != 0:
            return {"rc": -1, "path": "nvml"}
        h = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0:
            return {"rc": -1, "path": "nvml"}
        v = ctypes.c_uint()
        clk = v.value if lib.nvmlDeviceGetClockInfo(h, 2, ctypes.byref(v)) == 0 else None
        pw = ctypes.c_ulonglong()
        pwr = pw.value if lib.nvmlDeviceGetPowerUsage(h, ctypes.byref(pw)) == 0 else None
        tmp = ctypes.c_uint()
        tC = tmp.value if lib.nvmlDeviceGetTemperature(h, 0, ctypes.byref(tmp)) == 0 else None
        return {"rc": 0, "path": "nvml-ctypes", "clocks.sm": clk,
                "power.draw_mW": pwr, "temperature.gpu": tC}
    except Exception as exc:
        return {"rc": -1, "path": "nvml", "error": str(exc)}
