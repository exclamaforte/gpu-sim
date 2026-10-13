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

---

## The admission: which results were computed against the wrong part

Asked for explicitly, because a re-run is cheap and an admission is not.

**Triage result: no existing result in this lab depended on the three wrong
values. The exposure was LATENT, not active** — the same distinction
`gpu-sim-u2o.6.1.3`'s `cudaMemGetInfo` finding made earlier tonight.

- **No consumers.** A tree-wide search for `maxThreadsPerMultiProcessor`,
  `sharedMemPerMultiprocessor`, `l2CacheSize` and `occupancy` finds **no
  consumer** of these fields.
- `generalization_suites._occupancy_proxy` is a *proxy from block size*
  (`threads/block / 1024`) and hardcodes 1024 for `maxThreadsPerBlock` — which
  this fix confirms is the real 5090's value. It never reads the fake's fields.
- The fake exports `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, but nothing
  in the tree calls it.
- **17.x's PyTorch fixture was captured on the REAL 5090**: ts
  `20260927t163138-19bc8285` and predecessors, `torch.cuda.synchronize()` in
  the capture path, and a negative control that *blanks*
  `CUDA_VISIBLE_DEVICES` and gets `No CUDA GPUs are available; no fixture
  written`. A fixture that cannot be produced without a real device cannot have
  been produced by the fake.
- What *did* use the fake: only the `-cudart=none` smoke paths (6.1.3's
  `test_cudart_none.sh`, 8dw's preflight demo). None compute occupancy, sizes,
  or cache behaviour.

The general lesson, which is why this is worth stating rather than just
recording: **a fidelity bug in a simulator is invisible until something consumes
the field, and the consumer is usually code that has not been written yet.** The
cost of finding this now is low *precisely because* nothing had depended on it.

## Single-sourcing, and the channel bug it exposed

The PI's structural rule, adopted: a quantity reported through more than one
channel must be single-sourced, and the test must pin the **channels to each
other**, not one to a literal. My first test pinned the struct to hardcoded
numbers — which would have gone on passing while the attribute channel reported
something else entirely. That is exactly how the fake came to describe 2048
threads/SM through one channel and 1536 through another.

Pinning the channels found a **confirmed, larger bug**:

`shim/cuda_runtime.h` numbered `cudaDeviceAttr` 1, 2, 3, 4, 5, 6, 7, 10, 11,
75, 76. Real CUDA 13.3 numbers them 1, **39**, **16**, **8**, **12**, **13**,
**38**, 75, 76. Only `MaxThreadsPerBlock` (1), `ComputeCapabilityMajor` (75) and
`ComputeCapabilityMinor` (76) coincided.

So `cudaDeviceGetAttribute` returned **0 to every caller built against real CUDA
headers** for 3 of the 4 probed attributes — and the one that worked was exactly
the one whose number happened to match. After renumbering, **all** probed
attributes agree with the struct. I also deleted the `cudaDevAttrTotalGlobalMem`
case: that enumerator does not exist in CUDA 13.x, and keeping a case for it is
the same drift.

Device values are now defined **once**, as `FG_*` constants, and every channel
derives from them.

## Still open, better characterised

`maxThreadsPerBlock` and `warpSize` read 0 from the struct. With the attribute
channel now correct, the disagreement runs the **opposite** way to before — the
attribute says 1024, the struct says 0 — so **the struct path is the broken
one**, which reverses what the earlier evidence suggested.

Cause narrowed to: shim-struct fields are written *after* the overlay at offsets
that collide with it. I confirmed one collision — shim `totalL2Cache`@312 (8
bytes) overwrote the overlay's `pitch`@312 — and moving it before the overlay
fixed that one. **A second collision remains at 320.**

Reported as `WARN`, not asserted: failing the whole gate on an open item would
make it uninformative about the parts that *are* fixed. The next step is to
enumerate every shim-struct field landing in 288–384 and write them all before
the overlay, or drop the redundant shim assignments in favour of the overlay
entirely.
