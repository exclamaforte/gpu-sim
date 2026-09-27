# gpu-sim-9ti — `kernel_name` did not match the `__global__` in `source`

`generator_stage2..5` built every spec with `kernel_name=f"{op}_kernel"`
while their templates declare `__global__ void {op}(...)`. Nothing noticed
for a whole review cycle, because nothing had ever *compiled* those drivers.
The first thing that compiled them was gpu-sim-u2o.6.1.3's `-cudart=none`
requirement enumeration, and all 12 sampled specs failed with
`identifier "reduce_sum_block_kernel" is undefined`.

Any batch measurement over stage2-5 kernels would have failed at once, and the
failure would have looked like a CUDA toolchain problem rather than a
generator bug. This bead fixes it at the source and adds the executable
assertion that would have caught it.

## Two layers, because one is not enough

**1. The source is the authority.** `epic_u2o/kernel_names.py`:

* `declared_kernel_names(source)` — every `__global__` in a source.
* `resolve_kernel_name(spec)` — the name to launch, taken from the source.
  Raises only when the source declares no `__global__` at all (a genuinely
  broken spec) or when the source declares several and the dict matches none
  (refuses to guess).
* `check_kernel_name(spec)` — `None` when the dict agrees with its source,
  else a complaint naming both.

`kernel_driver.synthesize()` now launches `resolve_kernel_name(spec)`, so a
mismatched dict from a third party still produces a driver that **compiles**,
while a dict whose source has no kernel at all fails loudly instead of
emitting a call to nothing.

**2. The generators agree with their own output.** All 14 construction sites
in `generator_stage2..5` now use `_kernel_function_name(<the same string
passed as source>)`, so the mismatch is not representable.

## The test that would have caught it (CPU-only, ~90 s, no GPU)

`test_kernel_names.py` — 15 checks, all green:

* resolver/checker unit behaviour, including the multi-kernel "refuse to
  guess" case and the `extern "C" __global__ __device__` parse;
* for **every generator, stage 1 through 5**: 12 specs each, assert
  `kernel_name` is one of the `__global__`s in its source;
* and the part that actually matters: **each emitted driver is compiled with
  `nvcc`**, 60/60 across the five stages.

`negative_control.py` shows the defence in depth: a hand-built pre-fix spec
(`kernel_name='reduce_sum_block_kernel'`, source declaring
`reduce_sum_block`) is reported by `check_kernel_name` *and* still yields a
driver that launches the right function.

## Why no `ts` job

Proving a generated file compiles needs `nvcc`, not a GPU — that is the whole
point of this bead, and it is why the class of bug survived so long. Every
command here runs outside the queue and touches no CUDA device.

One honest note on the compile step: it uses `-cudart=shared`, not
`-cudart=none`. `-cudart=none` additionally requires libfakegpu to export the
whole `__cuda*` registration surface — gpu-sim-u2o.6.1.3's work, not yet in
master — and that link mode is covered by 6.1.3's own
`tests/test_cudart_none.sh`. Depending on it here would make this regression
test fail for an unrelated reason.

## Files

* `epic_u2o/kernel_names.py` (new) — the resolver and the checker
* `epic_u2o/generator_stage2..5.py` — 14 sites now derive the name from source
* `epic_u2o/kernel_driver.py` — `synthesize()` launches the resolved name
* `experiments/gpu-sim-9ti/test_kernel_names.py` — the executable assertion
* `experiments/gpu-sim-9ti/negative_control.py` + `.log`, `output.log`, `job.json`

`generator_stage1` was already correct and is covered by the same test.
