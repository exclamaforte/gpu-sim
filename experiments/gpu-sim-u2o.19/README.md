# gpu-sim-u2o.19 — the bead's framing is inverted on both counts

The bead said three deeper fields in the real-layout overlay "do not match what
the code intends" and that "the hardcoded byte offsets for these three are
wrong for this CUDA version". Both halves of that are wrong, in opposite
directions, and checking cost one CPU probe and one one-second GPU query.

## The offsets are correct

`offsetof()` against the real CUDA 13.3 headers, CPU-only:

| field | offset | size |
|---|---|---|
| `l2CacheSize` | 596 | 4 (an `int`, not a `size_t`) |
| `maxThreadsPerMultiProcessor` | 604 | 4 |
| `sharedMemPerMultiprocessor` | 624 | 8 |

`sizeof(cudaDeviceProp)` = 1008. The overlay's `memcpy` targets match exactly.
`l2CacheSize` being a 4-byte `int` is why the overlay must write an `int`; the
test pins that too.

## The VALUES are what was wrong

Read off the real 5090 (ts `20260928t105453-2dc7af7c`):

| field | code set | real 5090 | overstatement removed |
|---|---|---|---|
| `maxThreadsPerMultiProcessor` | 2048 | **1536** | 1.33× |
| `sharedMemPerMultiprocessor` | 233472 (228 KB) | **102400** (100 KB) | 2.28× |
| `l2CacheSize` | 134217728 (128 MB) | **100663296** (96 MB) | 1.33× |

The bead's own quoted "reads" numbers *were* the real ones. So the fake was
claiming a part with 2048 threads/SM and 128 MB of L2 that does not exist.

**Corrected in all three places** the value appears — the shim struct
assignment, the real-layout overlay, and `cudaDeviceGetAttribute`. If those
three disagree, the fake is self-inconsistent depending on which one a caller
happens to use, which is worse than either value being wrong.

## Why "low priority" undersells it
PyTorch reads `maxThreadsPerMultiProcessor` when it computes occupancy. An
occupancy-sensitive kernel sized against this fake would be sized against the
wrong part — and this epic's 17.x work is PyTorch traces.

## Still open, reported rather than hidden
`maxThreadsPerBlock` (overlay writes 1024 at byte 320, reads 0) and `warpSize`
(writes 32 at 308, reads 0) do not arrive, and `cudaDeviceGetAttribute`
returns 0 while the struct field is correct. **Cause undiagnosed.** The test
prints these as `WARN` rather than asserting them, because a gate that fails on
a defect I have not diagnosed trains people to ignore gates.

Offered as a **lead, not a conclusion**: the shim's attribute enum values may
differ from the real CUDA 13.3 ones, so a real-header caller can miss the
`switch` case entirely and get the default of 0. That is consistent with the
attribute failing while the struct fields are right, and inconsistent with the
overlay simply not running. Test the enum values before acting on it.

## A reproducibility finding, out of scope but material
A clean checkout of master **cannot build `libfakegpu.so`**, so `make check`
cannot run at all. Ten required files are untracked and not gitignored:
`src/fakegpu_{config,latency,nccl,pool,random}.c`, `include/fakegpu.h`,
`shim/nccl.h`, `tests/{fake_client,vector_add,ring_allreduce}.c`. Only
`src/fakegpu_cuda.c` and `shim/cuda_runtime.h` are tracked.

So every "make check PASS" in this lab — including the ones I and the reviewer
have run and quoted — holds only in a working tree that happens to have those
ten files. The `.gitignore` comment on `libfakegpu.so` already warns that a
missing library can make a gate pass **vacuously**; this is the stronger
version: the gate cannot start.

I have **not** fixed it here. Deciding what `src/`, `include/`,
`shim/nccl.h` and `tests/*.c` should track is **gpu-sim-u2o.6.11's** call — the
bead that deliberately decided the `epic_u2o/` tracking set. Reported rather
than silently widening my own bead's scope.
