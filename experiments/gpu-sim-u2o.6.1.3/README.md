# gpu-sim-u2o.6.1.3 — complete libfakegpu's cudart surface for `-cudart=none`

**No GPU was used, and no `ts` job was needed** (see `job.json`): the whole
bead is header/shim C plus `nvcc` compile+link, and the only binary executed
runs with `CUDA_VISIBLE_DEVICES=` blanked and links neither `libcudart` nor
`libcuda`.

## The problem, measured

`nvcc -cudart=none` on a generated driver did not link. Two measured gaps, in
the order they appear:

1. nvcc's generated host stub references `__cudaRegisterFatBinary`,
   `__cudaRegisterFatBinaryEnd`, `__cudaRegisterFunction`,
   `__cudaUnregisterFatBinary`, `__cudaPushCallConfiguration`,
   `__cudaPopCallConfiguration`, `__cudaGetKernel`, `__cudaLaunchKernel` —
   none of which libfakegpu exported.
2. Past those, the missing cudart API family: `cudaGetLastError`,
   `cudaPeekAtLastError`, occupancy/symbol queries, and the device
   error-state surface.

## What changed

* `shim/cuda_runtime.h` — declares the whole `__cuda*` registration/launch
  surface (signatures copied from the toolkit's `crt/host_runtime.h` and
  `crt/device_functions.h` so the ABI matches the real caller), plus
  `cudaGetLastError` / `cudaPeekAtLastError` / `cudaThreadExit` / occupancy /
  symbol queries. Also:
  * `uint3` and `cudaKernel_t` typedefs the new prototypes need;
  * `struct CUstream_st;` forward-declared at file scope — a struct first
    named *inside a parameter list* gets prototype scope, so the second
    declaration naming it is a different type and gcc rejects the pair
    (this bit me and is worth remembering);
  * the false claim that the fake path "runs GPU-free under LD_PRELOAD" is
    **deleted** and replaced with the measured truth: nvcc defaults to
    `-cudart=static`, so cudart's symbols live inside the executable and
    `LD_PRELOAD` cannot interpose them.
* `src/fakegpu_cuda.c` — implements the surface. Registration is bookkeeping,
  so no-ops returning success/0 are the correct fake behaviour; a launch goes
  through the same `fg_validate_launch()` + `fg_simulate_kernel()` path as
  `cudaLaunchKernel`. Also adds a **sticky error state**: nvcc's stub
  *discards* the return value of `__cudaLaunchKernel`, so without it an
  invalid launch config vanished silently and the preflight smoke test would
  have approved a broken program. `cudaGetLastError` reports-and-clears,
  `cudaDeviceSynchronize` surfaces a pending failure — as real cudart does.

## Evidence, not inspection

`tests/test_cudart_none.sh` (wired into `make check`, which is green) asserts,
mechanically:

| # | assertion | result |
|---|---|---|
| 1 | `nvcc -cudart=none … -lfakegpu` links | PASS |
| 2 | `ldd` matches neither `libcudart` nor `libcuda`; `libfakegpu` is the provider | PASS |
| 3 | runs with `CUDA_VISIBLE_DEVICES=`, exit 0, 3 trial JSON lines, fake's configured latency (2 ms) | PASS |
| 4 | an invalid launch config is **rejected** by the fake | PASS |
| 5 | the same compile **without** `-lfakegpu` fails to link | PASS — so the fake is the only provider, no silent fallthrough |
| 6 | all new symbols in `nm -D --defined-only libfakegpu.so` | PASS |

Assertion 5 is the one that matters for the "no symbol may fall through to a
real CUDA library" rule: if libfakegpu were missing anything, the negative
build would silently start picking symbols up from a real cudart instead of
failing.

`experiments/gpu-sim-u2o.6.1.3/enumerate_requirements.py` enumerates the real
requirement instead of guessing: it renders specs from **stage generators 1–5**
through `kernel_driver.synthesize()` and compiles each with `-cudart=none`.
Result (`metrics.json`): **15/15 drivers link, zero missing symbols**, plus
the canonical stage-1 harness driver.

## Bug found en route (filed as `gpu-sim-9ti`)

All 12 stage-2..5 specs sampled have a `kernel_name` that does not match the
`__global__` function in `source` (`reduce_sum_block_kernel` vs
`__global__ reduce_sum_block`, and likewise for stages 3/4/5), so their
synthesized drivers fail to *compile*. That is a generator defect, not a shim
gap; the enumeration script works around it locally purely so the cudart
surface could be measured, and that workaround is explicitly not a fix.

## Known limitation, stated rather than hidden

`FAKEGPU_KERNEL_LATENCY_MS` is parsed with `parse_long`, so a fractional
value (e.g. the `0.5` the harness's preflight sets) silently becomes 0 and the
fake reports `time_ms: 0.000000`. The test uses an integer latency. Changing
the config field to a double is a wider change than this bead, so it is
reported, not silently patched.

Device-memory *contents* remain fake-by-design: `**global**` bodies never
execute in an API-surface fake. Only API traffic, launch validation and
timing are modelled.
