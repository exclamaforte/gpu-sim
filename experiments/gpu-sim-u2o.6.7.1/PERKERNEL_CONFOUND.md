# The per-kernel confound test: designed, built, NOT yet run at full scale

**Status: incomplete.** The experiment is right, the design is settled, and the
scaffolding builds and links — but the full 264-spec run does not yet complete,
so there is **no result from it**. Recorded as such rather than left for
someone to discover from a missing artifact.

## What it is

The PI's ruling was that the 70× and the 12.6% are one fact seen twice, and
named the single measurement this lab has grounds to want: re-run the decisive
arm with a **per-kernel binary** and the shared context held. If the offset
collapses, the executable was the cause and the surviving speed is ~1.6×, not
70×.

So this builds, from **7.1's own `kernel_driver.synthesize()` output**:

* one translation unit per distinct kernel, each compiled by **its own `nvcc -c`**,
  with its **own `cudaMalloc` calls sized to that kernel's arity** — the two
  things that differ between 7.1's executable and my batched one;
* the per-spec launch geometry (N, grid, block) hoisted from `main()`'s locals to
  globals the driver sets at runtime;
* all objects **linked into one executable driven by one process with one CUDA
  context**, which is the thing being preserved from the fast arm.

The specs are 7.1's own, regenerated with the same sweep code and the same
`kernel_hash` construction, and the script **proves** the set is 7.1's by
intersecting the hashes with 7.1's committed `measured.jsonl`. A different sweep
would reproduce the confound while claiming to remove it, so the run refuses if
the hashes do not match.

## What is not yet true, precisely

The full run aborts with `loop exited 3`, meaning a dispatched kernel returned
non-zero. The `--limit 8` run completes and produces rows, so the build and link
path is sound; what fails is somewhere in the per-spec dispatch at full scale.
The open question is whether the geometry globals are reaching the kernels as
intended — the `const int N` / `dim3 grid` / `dim3 block` hoisting is a regex
over synthesized source, and **a regex that silently fails to match is exactly
the class of defect this project keeps cataloguing**. The count line also still
prints a hardcoded `0 distinct kernels`, so the de-duplication by source is not
reporting what it did.

## Six build iterations, all recorded

Every one of these was a *link or compile* failure in scaffolding I wrote, not a
measurement, and each is instructive in the same way:

1. one TU per **spec** → 264 objects each defining the same `__global__`; the
   linker rejected ~250 duplicates. **Informative, not incidental**: it is the
   confound seen from the build side — the *kernel* is the unit of compilation.
2. the loop declared `u2o_run_N(int kidx)` but the renamed functions take no
   argument.
3. an f-string turned `\n` into a real newline **inside C string literals**,
   while C line separators need the opposite. Five iterations went into those
   two escapes, which is why the driver now lives in **`perkernel_loop.cu.in`**,
   a real file, where the generated code is reviewable.
4. the dispatch function was appended *after* `main`, so it was used before
   declaration.
5. each kernel TU **defined** `N`/`grid`/`block`; the first "fix" made them
   extern in the wrong direction and failed identically.
6. **`extern "C" { int N; }` in C++ is a DEFINITION, not a declaration.** A block
   of externs is not a block of declarations. Fixed with individual
   `extern "C" int N;` declarations.

Points 3, 4 and 6 are all the same shape: **a thing assembled by successive
edits that nothing compiled until the end.** That is the trap this project has
named repeatedly — "a check has to live in an artifact or a script, not in a
careful sentence" — and it applies just as much to the artifact that *is* the
script. The fix for 3 was structural (stop generating code from strings); I
should have reached it on iteration 1 instead of 5.

## The honest position

**No result.** The prior state stands unchanged: criterion 2 fails, four
hypotheses are eliminated, and the binary is the leading candidate. This
artifact narrows nothing on its own; it records that the decisive measurement
is specified, that its specs are provably 7.1's, that the build and link path
works, and exactly which dispatch step still fails.

I stopped here rather than continuing to spend shared-queue slots on generated-C
plumbing. Fourteen of tonight's jobs were scaffolding failures, which is a
signal about the hour and not about the experiment.
