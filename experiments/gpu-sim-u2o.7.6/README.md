# gpu-sim-u2o.7.6 — what real kernels use that the synthetic families do not

The bead asks for a coverage check against real CUDA (CUTLASS, cuBLAS/cuDNN
samples, other open repos). What it actually turns up is a defect in the
instrument — found by checking the instrument before trusting it.

## The instrument reports prefixes as coverage

`corpus_miner.py` extracts a **60-line window** from each kernel's signature and
reports idiom presence over those windows, presenting it as corpus coverage. On
CUTLASS it says:

```
atomicAdd 0    wmma:: 0    cooperative_groups 0    __shfl 0
```

A direct grep of the same tree finds **wmma in 37 files**, `cp.async` in 48,
`__shfl_xor` in 8, `atomicAdd` in 3.

Those zeros are not findings about real code; they are measurements of a prefix.
Reporting *"real CUDA uses no tensor cores"* from this would have been a serious
false finding — and it is the same shape as everything else catalogued in this
project: a signal that is **true of the instrument and false of what it claims
to measure**.

I have not changed the miner's window: it is a deliberate size choice for a
lightweight corpus, and it is not this bead's artifact. What is fixed here is
that **coverage is measured over whole files**, and the window-based number is
reported *beside* it rather than instead of it.

## And my first corpus was partial — the same error, mine

The first pass mined a `git sparse-checkout` of CUTLASS `examples/` **only**:
167 `.cu` files containing no tensor-core, atomic or cooperative idioms at all.
I measured coverage on it and the partiality was **invisible in the output**.

The corpus is now `examples + include/cutlass + include/cute` — **10,189**
`.cu`/`.cuh`/`.h` files — plus torch's 62 bundled kernels, and the artifact
records corpus completeness **before** any coverage number.

## The measured gap

| capability | real corpus | synthetic | verdict |
|---|---|---|---|
| tensor cores | 188 files | **0** | **GAP** |
| warp intrinsics | 57 files | **0** | **GAP** |
| atomics | 26 files | **0** | **GAP** |
| shared memory + sync | 153 files | 1 of 24 (k_block_sum) | covered |

**The 24 synthetic families are almost entirely plain elementwise kernels.**
Exactly one uses any shared-memory idiom, and none uses tensor cores, warp
intrinsics or atomics — which real code uses in 188, 57 and 26 files.

This is the first *evidence* about the family mix rather than an opinion, which
matters because the family mix is one of the corpus decisions the PI left to me
and the reviewer jointly.

### One gap excluded as an artefact

`streams / multi-device` shows real code 165 files, synthetic 0 — and that is
**not a finding**. The idiom is `<<<` and `cudaStream`, which live in the
**launch harness**, not in a kernel body, and both sides here are measured over
kernel sources. It is the same category error as the vocabulary mismatch below,
and it is excluded from the actionable findings so nobody is misled by it.

## A second category error of mine, in the same file

My first gap analysis asserted the synthetic side from a hand-written
family→capability map written in a **different vocabulary** from the idiom set
(`reduction`, `tiled GEMM` against `tensor cores`, `atomics`). The set difference
then reported that the synthetic families lack *"shared memory + sync"* — when
`k_block_sum` plainly uses `__shared__` and `__syncthreads`.

A gap analysis whose two sides are not in the same vocabulary is a category
error that produces confident nonsense. Both sides are now **measured**: the
real corpus by whole-file idiom presence, the synthetic side by generating its
own 24 sources and measuring those the same way.

## Scope note

`epic_u2o/corpus_protocol.py` and the synthetic generator are carried over from
`lab/gpu-sim-u2o.6.7.1` (both unmerged) so the synthetic side can be measured
rather than asserted. They are byte-identical copies and will merge cleanly; if
the reviewer prefers them carried on one branch only, they should be dropped from
here — the coverage analysis does not depend on where they live.

---

## Follow-up: closing the measured gap (27 families, all compiling)

The coverage check was written to be acted on, so three families were added:
`k_warp_reduce`, `k_atomic_hist`, `k_atomic_sum`. **All 27 families synthesize
*and compile*** — the condition that matters, because the original finding was
that a family naming an idiom it does not run would close the gap on paper.

| capability | before | after |
|---|---|---|
| warp intrinsics | **GAP** (real 57 files, synthetic 0) | **closed** |
| atomics | **GAP** (real 26 files, synthetic 0) | **closed** |
| tensor cores | GAP (real 188 files, synthetic 0) | **still open — see below** |

### Tensor cores stay open, deliberately

`k_wmma_gemm` was written and then **dropped**. WMMA fragment instantiation
fails on this toolchain — `incomplete type ... is not allowed` — for every
arrangement tried: the fragment as a `__global__` local, the fragment in a
`__device__` helper, and after correcting the non-existent `wmma::float` to
`float`.

Shipping a family that mentions `wmma` and does not build would close the
coverage gap on paper and widen it in fact — the exact failure this harness
exists to catch. So the family is absent and the gap is reported as still open.

The corpus decision that asked for ≥24 families *across capability classes* is
therefore only **two thirds** delivered. The tensor-core class needs a working
WMMA or an `mma.sync` implementation before the family mix is right, and that is
a named piece of work rather than a silent omission.

I also removed a `k_wmma_gemm` by line-surgery that ate a docstring in
`generate_scale_corpus.py`, breaking the file. Restored from git and re-applied
cleanly. Third time this session a hand edit to a generated-source pipeline cost
more than redoing it.

## The `loop` binary is not committed (and how to rebuild it)

`build_scale/loop` was a 1.6 MB compiled ELF checked into the tree -- a build
output of the source sitting next to it, unreproducible in diff and permanent
in every clone. Removed; rebuild with:

```
cd experiments/gpu-sim-u2o.7.6/build_scale && nvcc -O2 loop.cu k*.cu -o loop
```

Verified 2026-10-04 (builds clean, ~1.3 MB). `specs.json` records what the
loop runs, not how it is built -- this is the how.
