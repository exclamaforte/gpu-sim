#!/usr/bin/env python3
"""CPU-only regression test for gpu-sim-9ti (kernel_name vs source).

NOTE on the compile step: it uses `-cudart=shared`, not `-cudart=none`. This
test is about whether the GENERATED SOURCE compiles, which needs no CUDA
runtime at all. `-cudart=none` additionally requires libfakegpu to export the
whole __cuda* registration surface -- gpu-sim-u2o.6.1.3's work, not yet in
master -- and is covered by 6.1.3's own tests/test_cudart_none.sh. Using it
here would make this regression test fail for an unrelated reason.

gpu-sim-9ti existed for a whole review cycle because the mismatch lived in a
f-string nobody executed. This asserts, for every generator and every
generated spec, that:

  1. spec['kernel_name'] is one of the __global__ functions in spec['source'];
  2. kernel_driver.synthesize(spec) launches a name that IS declared in the
     emitted source (the compiler is the judge, not this test);
  3. the emitted driver actually compiles with nvcc -cudart=none.

(3) is CPU-only -- nvcc is a compiler, and compiling touches no GPU. That is
what caught the original bug: the stage2-5 drivers did not compile. The GPU
is not needed to know that, so neither is it needed to keep them compiling.

No GPU, no torch, no ts queue: python3 experiments/gpu-sim-9ti/test_kernel_names.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]   # experiments/<id>/ -> worktree root
sys.path.insert(0, str(REPO))

from epic_u2o.kernel_driver import synthesize  # noqa: E402
from epic_u2o.kernel_names import (  # # noqa: E402
    check_kernel_name, declared_kernel_names, resolve_kernel_name,
)

STAGES = ["generator_stage1", "generator_stage2", "generator_stage3",
          "generator_stage4", "generator_stage5"]
PER_STAGE = 12          # enough to cover the op families, cheap to compile
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + str(detail)[:150] if detail else ''}")
    if not ok:
        FAILS.append(name)


# --- 1. unit behaviour of the resolver --------------------------------------
good = {"source": "__global__ void k(float* a){ }\n", "kernel_name": "k"}
check("resolver: matching name is used", resolve_kernel_name(good) == "k")
check("checker: matching name is clean", check_kernel_name(good) is None)

mismatched = {"source": "__global__ void real_name(float* a){ }\n",
              "kernel_name": "fake_name_kernel"}
check("resolver: source wins over a mismatched dict",
      resolve_kernel_name(mismatched) == "real_name")
complaint = check_kernel_name(mismatched)
check("checker: mismatch is reported, not silent",
      complaint is not None and "fake_name_kernel" in complaint, complaint)

empty = {"source": "int main(){ return 0; }", "kernel_name": "whatever"}
try:
    resolve_kernel_name(empty)
    check("resolver: source with no __global__ raises", False, "no exception")
except ValueError as e:
    check("resolver: source with no __global__ raises", "no __global__" in str(e), str(e)[:90])

ambiguous = {"source": "__global__ void a1(float* x){ }\n__global__ void b2(float* x){ }\n",
             "kernel_name": "c3"}
try:
    resolve_kernel_name(ambiguous)
    check("resolver: refuses to guess with several kernels", False, "no exception")
except ValueError as e:
    check("resolver: refuses to guess with several kernels", "refusing to guess" in str(e),
          str(e)[:90])

check("parser: sees templated/inline qualifiers",
      declared_kernel_names(
          '__global__ void f1(const float* a){ }\n'
          'extern "C" __global__ __device__ void f2(int* b){ }\n') == ["f1", "f2"])

# --- 2. every generator, every spec, and the compiler ------------------------
tmp = Path(tempfile.mkdtemp())
for stage in STAGES:
    mod = __import__(f"epic_u2o.{stage}", fromlist=["generate_kernels"])
    specs = []
    for spec in mod.generate_kernels():
        specs.append(spec)
        if len(specs) >= PER_STAGE:
            break
    if not specs:
        check(f"{stage}: produced specs", False, "generator yielded nothing")
        continue

    bad_name = [s.to_dict()["kernel_name"] for s in specs
                if check_kernel_name(s.to_dict())]
    check(f"{stage}: kernel_name matches source for {len(specs)} specs",
          not bad_name, f"mismatched: {bad_name[:3]}")

    compiled = 0
    compile_errors = []
    for i, s in enumerate(specs):
        d = s.to_dict()
        try:
            program = synthesize(d)
        except Exception as e:
            compile_errors.append(f"synthesize: {e}")
            continue
        # the emitted driver must launch a name it declares
        declared = declared_kernel_names(program)
        if d["kernel_name"] not in declared and len(declared) == 1:
            compile_errors.append("emitted driver launches an undeclared name")
            continue
        cu = tmp / f"{stage}_{i}.cu"
        cu.write_text(program)
        r = subprocess.run(
            ["nvcc", "-cudart=shared", "-arch=sm_120", "-O2", str(cu),
             "-o", str(tmp / f"{stage}_{i}.out")],
            capture_output=True, text=True, timeout=600)
        if r.returncode == 0:
            compiled += 1
        else:
            head = [l for l in r.stderr.splitlines() if "error" in l][:2]
            compile_errors.append("; ".join(head))
    check(f"{stage}: {compiled}/{len(specs)} drivers compile",
          compiled == len(specs), compile_errors[:2])

print()
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
