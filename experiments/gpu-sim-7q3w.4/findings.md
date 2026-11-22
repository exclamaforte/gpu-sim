# gpu-sim-7q3w.4 -- first measurements: the two risks the bead says to measure first

Nothing here is a model, a launcher or a claim of success. This is the capacity and
API-surface survey the bead asks for BEFORE any of that is built, because both
answers change the design.

    python3 experiments/gpu-sim-7q3w.4/probe_capacity.py           # both surveys
    python3 experiments/gpu-sim-7q3w.4/probe_capacity.py --with-fork  # + the dead end
    python3 experiments/gpu-sim-7q3w.4/probe_capacity.py --json

## 1. Capacity (risk a): the 8-node design fits, but only on a quiet box

Measured with the same probe, twice, ~25 minutes apart on the same machine:

| measurement | value |
|---|---|
| MemTotal | 31117 MiB |
| MemAvailable, first sample | **4224 MiB** (another agent held ~10.8 GiB) |
| MemAvailable, second sample | 8991 MiB |
| child: `import torch` peak RSS | 543 MiB |
| child: torch + 12M-param transformer + AdamW step | **970 MiB** |
| 8 nodes x 970 MiB | 7760 MiB |

So the bead's estimate (~0.7 GiB/node) was low by ~40% once a real model and
optimizer are resident: the design needs **7.8 GiB**, which fits the second sample
(1.2 GiB spare) and does **not** fit the first one at all. The consequence is a
design requirement, not a note: the launcher needs a RAM preflight (like
`scripts/gpu_run.sh`'s quota preflight) that refuses with the measured numbers
rather than letting 8 ranks get OOM-killed mid-run -- and the evidence bundle must
record MemAvailable at start, because "it worked" is otherwise a statement about
what else happened to be running.

## 2. fork/COW is a measured dead end (do not design around it)

The obvious way to dodge 8x torch import is one warmed parent that forks 8 ranks:
copy-on-write would share the 543 MiB of imported pages. Measured: it **deadlocks**.

    DeprecationWarning: This process (pid=...) is multi-threaded, use of fork() may
    lead to deadlocks in the child.
    -> no child returns; 90s timeout, no exit codes printed.

CPython 3.14 says it before starting, and it is right: torch has thread pools up, and
the forked children never reach `backward()`. Kept behind `--with-fork` so the
finding is reproducible instead of folklore. **The 8-node design pays 8 full
imports**; that is what makes section 1 matter.

## 3. Driver-API surface (risk c): torch does not run on the current fake driver

`libtorch_cuda.so` has **zero undefined `cu*`/`cuda*` symbols**, so the intuitive
instrument -- "diff torch's undefined CUDA symbols against our shim's exports" --
measures nothing: torch resolves the driver API dynamically (`cuGetProcAddress` /
`dlopen("libcuda.so.1")`). The runtime answer is the one that counts, and it is:

    torch with libfakegpu.so installed as libcuda.so.1:
      device_count 1
      is_available False
      get_device_name(0) -> RuntimeError: Found no NVIDIA driver on your system...

libfakegpu exports 232 symbols, but the driver entry points torch needs are simply
absent: `cuInit`, `cuCtxCreate_v2`, `cuMemAlloc_v2`, `cuMemGetInfo`,
`cuGetProcAddress` are all missing (it was built for NCCL's device queries, not for
a framework's driver use).

Consequence for D3: "per-rank device context over D1's properties" is a **much**
larger shim than the NCCL work needed -- a driver-API layer with contexts,
allocations, streams and `cuGetProcAddress` version negotiation -- or the compute
split the bead already describes has to be made literal: torch tensors on the host
(real float32 math, 2 CPUs), with the collectives' device buffers living in our
pool and moved by real NCCL. The second is the cheaper honest route to D4/A5 and
the one the bead's COMPUTE paragraph already implies; the first is what "torch
believes it is on a GPU" (A1) literally requires. That trade-off is a PI decision,
not an implementer one, so it is recorded here rather than assumed.

## 4. What is NOT done

D1 (specs/b300_8x8.json), D2 (node illusion), D4 (the model), D5 (launcher) and the
A1-A8 assertions are all still open. No metrics.json is written yet: this increment
makes no claim that a gate could cite.

## Reproduce

    python3 experiments/gpu-sim-7q3w.4/probe_capacity.py --json

The probe imports nothing heavy itself (every torch measurement runs in a child, so
the parent's RSS cannot contaminate the number) and writes no files outside a
`.lab/tmp` scratch directory that it removes.
