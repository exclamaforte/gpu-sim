# The sampling increment, and what it revealed

`6.7.1`'s own attribution said the dominant remaining cost was the per-example
sample at ~39 ms — the `nvidia-smi` subprocess. This is the increment that acts
on that, and it is a continuation of the same bead rather than a new one.

## What changed

`nvidia-smi` per row replaced with **in-process NVML via ctypes** — the same
library `nvidia-smi` calls, loaded once, so a query is a function call instead of
a fork/exec plus a driver handshake. Measured: **0.0286 ms per sample against
~39 ms**, a ~1,370x cheaper sample. A `nvidia-smi` fallback remains and **every
row records which path produced its sample**, so a reader never has to guess.

| | ex/s | per hour |
|---|---|---|
| 7.1 single-run | 2.39 | 8,599 |
| batched, nvidia-smi sampling | 25.05 | 90,169 |
| **batched, NVML sampling** | **168.81** | **607,714** |
| ceiling (UPPER BOUND) | 7,911 | — |

**70.7x** over the single-run path. Protocol
unchanged: 40 timed launches per example, 2 invocations x 20, 264/264 distinct
per-example sample times, one process.

## Two bugs I hit doing it, both invisible in the code

1. **A segfault, twice.** `nvmlDeviceGetMemoryInfo` takes a 5-field struct, and
   I passed a 4-byte `c_uint` for it, so NVML wrote 40 bytes into a 4-byte slot.
2. **The same segfault again, for a different reason.** My helper was
   `_u32(fn, arg)` calling `fn(arg, byref(v))` — correct for two-argument calls,
   but the NVML getters take `(handle, type, out)`, so I was passing the sensor
   type *as the handle* and a pointer *as the type*. Neither returned an error;
   both corrupted the stack.

The general point: **a wrong FFI call does not fail, it corrupts.** Both bugs
were found by running, and both would have been invisible to a code review —
which is an argument for the cheap preflight habit rather than for care.

## What it revealed: the bottleneck MOVED, and it is now queue_state()

The attribution after this increment:

| component | ms/example | share |
|---|---|---|
| device work | 0.139 | 2.3% |
| **queue_state()** | **5.54** | **93.5%** |
| residual | 0.25 | 4.2% |
| total | 5.92 | |

The NVML read is 0.029 ms. The remaining 5.54 ms
is `queue_state()`, which calls `_load_all()` — it re-reads **every** job file in
the queue directory, about 500 of them, once per example. Measured directly:
**6.09 ms per call, x264 rows = 1.61 s, which is the entire run.**

So the bottleneck has moved again, and the next increment is bounded by
measurement rather than guesswork: `queue_state()` needs a short TTL cache or
should read only the running set. That is a change to a 6.6.6-owned module, so
it belongs on its own bead rather than here.

## Criterion 2 still fails, and slightly worse

189/264 medians reproduce, median
relative difference 15.6% — against
189.0 of 264 at the slower
25.05 ex/s run.

There is a suggestive pattern and I am not overclaiming it: the faster the loop,
the WORSE the reproduction. 25.05 ex/s gave a 12.5% median deviation; 168.81
ex/s gives 15.6%. Two points is not a trend, but it points the same way as the
warm-state hypothesis — a tighter loop leaves more L2 and TLB state carried
between examples, which is exactly the kind of effect that makes a batching fix
change labels. Separating that from the shared-CUDA-context effect still needs
the run that varies one at a time.
