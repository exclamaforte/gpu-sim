# gpu-sim-8dw — the preflight that could not fail, now repaired

`PreflightValidator._run_under_fakegpu` was:

```python
return True, "Preflight passed (placeholder)", ""
```

Unconditional. The fake-GPU smoke test did not happen and reported success,
and `safe_harness.preflight_check()` reaches this module.

## The defect, demonstrated rather than described

`demo_before_after.py` shows **both sides** and is self-contained: the pre-fix
module is committed beside it as `pv_pre_fix.py`, so there is no `/tmp`
dependency and a reader can re-run it from a clean checkout.

```
BEFORE  a kernel that does not exist  -> passed=True   'Preflight passed (placeholder)'
BEFORE  an empty launch config        -> passed=True   'Preflight passed (placeholder)'
AFTER   a kernel that does not exist  -> passed=False  'invalid launch config: block 999999x1x1 ... (max 1024)'
AFTER   an empty launch config        -> passed=True   'launch_done grid=(1,1,1) block=(1,1,1) failures=0'
AFTER   a good kernel (end to end)    -> passed=True   evidence: launch_done present
AFTER   cannot check (no libfakegpu)  -> passed=False  "cannot run ... 'cannot check' is NOT a pass"
```

Note the third row honestly: the repaired module **passes** the empty-config
probe, because an empty config defaults to a legal 1×1×1 launch and the
kernel really does run. That is correct behaviour, and it is also a claim I got
wrong first time — a draft of this file said the repaired module "answers
False to both probes", which its own output contradicts. The sentence was
written before the data existed.

## What it does now

Builds a harness `.cu` that pastes in the kernel under test, launches it with
the **configured** grid/block, and checks every CUDA return value — context
creation, both `cudaMalloc`s, both `cudaMemset`s, `cudaGetLastError` for
launch-time errors, `cudaDeviceSynchronize` for execution errors, and the
frees. It links with `-cudart=none` against libfakegpu directly rather than
`LD_PRELOAD`, because nvcc defaults to `-cudart=static` and a preloaded symbol
is shadowed by the one inside the executable (measured on this box by 6.1.3,
not assumed).

**The branch that matters:** if libfakegpu cannot be found, the check returns
**False** with "cannot run … and 'cannot check' is NOT a pass". "Not
implemented" and "passed" must never be the same answer.

## Two more defects the stub was hiding

1. `_compile_kernel` invoked `nvcc` **without `-c`**, so it tried to *link* a
   translation unit containing only `__global__` functions and always failed
   with `undefined reference to 'main'`. Step 1 of `validate_kernel` could
   therefore never succeed, and callers saw a compile failure instead of the
   smoke-test result they asked for. Now compiles with `-c`.
2. **libfakegpu did not export `cudaMemGetInfo`** — and 6.3's
   `kernel_driver.py` emits a call to it, so any `-cudart=none` link of a
   6.3-generated driver failed with `undefined reference to cudaMemGetInfo`.
   The smoke test tripped over the same gap. Exported now, reporting the
   fake's configured device memory.

## Test

`tests/test_preflight_cannot_pass.py` — 7 checks, CPU-only. Every check is
about **failure**, because a happy-path test would have passed against the
stub forever:

| check | result |
|---|---|
| a good kernel passes | PASS |
| …and carries evidence (`launch_done`, `failures=0`) | PASS |
| a kernel with an undefined symbol is rejected | PASS |
| an out-of-range launch config is rejected (`block 2048 > 1024`) | PASS |
| **"cannot check" FAILS rather than passing** | PASS |
| …and the message says why, in those words | PASS |

`make check` (6.1.3's `-cudart=none` suite, which touches the same
`src/fakegpu_cuda.c`) was re-run after the change: **9/9 PASS**.
