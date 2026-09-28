#!/usr/bin/env python3
"""CPU-only regression test for gpu-sim-8dw.

_preflight_validator.PreflightValidator._run_under_fakegpu used to be:

    return True, "Preflight passed (placeholder)", ""

-- a validation that could not fail, reachable from
safe_harness.preflight_check. A test that only checks the happy path would
have passed against that stub forever, because the stub passes everything.

So the checks here are all about FAILURE:

  1. a kernel that does not compile must be reported as a failure;
  2. an out-of-range launch config must be reported as a failure;
  3. "cannot perform the check" (no libfakegpu) must FAIL LOUDLY, not pass --
     this is the assertion the stub could never satisfy;
  4. a good kernel passes AND carries evidence (it really launched);
  5. the module can no longer be tricked into a pass by a kernel that
     references an undefined symbol at launch time.

CPU-only: the smoke path compiles with -cudart=none and links libfakegpu, so
the run needs no device. CUDA_VISIBLE_DEVICES is NOT blanked as a "sandbox",
because a device-free claim must not be manufactured by hiding the GPU.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[0]          # tests/ -> worktree root
sys.path.insert(0, str(REPO))

FAILS = []

GOOD = """
__global__ void add_k(const float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] + 1.0f;
}
"""

UNDEFINED_SYMBOL = """
__global__ void bad_k(const float *a, float *b, int n) {
    b[0] = no_such_function_exists_anywhere;
}
"""


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}"
          f"{'  ' + str(detail)[:150] if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    from epic_u2o.preflight_validator import PreflightValidator

    v = PreflightValidator(arch="sm_120")
    check("libfakegpu is resolvable for the smoke path", v.fakegpu_available)

    ok_cfg = {"grid_x": 8, "block_x": 128, "problem_size": 4096}

    r = v.validate_kernel(GOOD, ok_cfg, "add_k")
    check("1. a good kernel passes", r.success and r.run_ok, r.errors[:1])
    check("4. the pass carries EVIDENCE (it really launched)",
          "launch_done" in (r.stdout or "") and "failures=0" in (r.stdout or ""),
          (r.stdout or "").strip()[-80:])

    r2 = v.validate_kernel(UNDEFINED_SYMBOL, ok_cfg, "bad_k")
    check("5. a kernel with an undefined symbol is rejected",
          not r2.success, r2.errors[:1])

    r3 = v.validate_kernel(GOOD, dict(ok_cfg, block_x=2048), "add_k")
    check("2. an out-of-range launch config is rejected",
          not r3.success and "invalid launch config" in str(r3.errors),
          str(r3.errors)[:110])

    # 3. THE ONE THE STUB COULD NOT SATISFY: unable to check != passed.
    v2 = PreflightValidator(arch="sm_120", fakegpu_lib="definitely-not-here.so")
    r4 = v2.validate_kernel(GOOD, ok_cfg, "add_k")
    check("3. 'cannot check' FAILS rather than passing",
          not r4.success, str(r4.errors)[:130])
    check("3. and the message says why, in those words",
          "NOT a pass" in str(r4.errors) or "cannot run" in str(r4.errors),
          str(r4.errors)[:110])

    print()
    print(f"{len(FAILS)} failure(s)"
          + (": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
