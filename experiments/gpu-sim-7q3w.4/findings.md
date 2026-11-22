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

## 5. The driver shim: the device illusion works, torch does not bind to it yet

Per the PI ruling (virtual devices alias the real GPU; identity modelled, arithmetic
real) the shim is `shim/vdev.c` + generated pass-through trampolines
(`shim/gen_forwarders.py`): 23 overrides (enumeration, identity, B300 memory figure,
primary contexts) and 714 generated trampolines for every other `cu*` the real driver
exports, so no hand-written prototype can silently corrupt a call.

**Measured, working:** with the shim on `LD_LIBRARY_PATH` a direct driver-API caller gets
exactly the illusion --

    cuInit -> 0;  cuDeviceGetCount -> 8;  cuDeviceGetName(0) -> "NVIDIA B300"

**Measured, NOT working:** `import torch; torch.cuda.device_count()` under the same
environment reports **1** and `NVIDIA GeForce RTX 5090`. The shim is not in the path at
all; the real driver is. Four failures found the hard way, each now documented in code:

1. our file named `libcuda.so.1` with DT_SONAME `libcuda.so.1` makes `dlopen` of the
   real driver return *ourselves* -> infinite `cuInit` recursion (gdb: 12 frames);
2. loading the real driver with `RTLD_GLOBAL` lets its internal PLT calls resolve to our
   symbols -> same recursion;
3. `dlmopen(LM_ID_NEWLM, ...)` gives a clean namespace but the NVIDIA driver crashes in
   its own mutex there -- not an option;
4. exporting a trampoline for `cuGetExportTable` makes libnvidia-ml call our jump into a
   dispatcher that is not meant to be reached that way (SIGSEGV inside `nvmlInit`); the
   generator now refuses to export it.

**CORRECTION (PI measurement 2026-11-22, supersedes this section's conclusion):**
torch does **not** link cudart statically.  `readelf` on libtorch_cuda.so shows
`cudaGetDeviceCount` imported and `NEEDED libcudart.so.13`, and the deciding call is
neither libcuda nor cudart: **NVML wins**.  The measured chain in one process is

    torch.cuda._device_count_nvml()  -> 1     <- the decider (pre-init)
    torch._C._cuda_getDeviceCount()  -> 8     <- dynamic cudart interposition binds
    torch.cuda.device_count()        -> 1     <- NVML wins
    torch.cuda.init()                -> AcceleratorError 'invalid device ordinal'

The driver shim is still correct for direct driver-API callers and for the NCCL/NVML
paths later; it is simply not the layer torch enumerates through.  The libfakecuda
export-table route is not used either: torch's enumeration is NVML + cudart.

## 6. What binds torch: NVML count, cudart properties, and the forwarder SIGSEGV

The two layers the PI specified, plus the real cause of the probe's child SIGSEGV:

**Layer 1 -- NVML (`shim/nvml.c`, `shim/gen_nvml_forwarders.py`).**  Installed as
`libnvidia-ml.so.1` on `LD_LIBRARY_PATH` (pynvml dlopens and closes it, so a
preloaded different soname is not enough).  Modelled: count 8, name, UUID, PCI bus id,
288 GiB total, PCIe gen/width.  Forwarded: real free bytes, real compute capability,
driver version, and the rest of the 400+ entry points.  Measured alone: `torch.cuda.device_count() -> 8`.

**Layer 2 -- cudart (`shim/cudart.c`), LD_PRELOAD.**  Overrides `cudaGetDeviceCount`,
`cudaSetDevice`, `cudaGetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`,
`cudaDeviceGetPCIBusId`, `cudaMemGetInfo`; every virtual index aliases real device 0.
Measured: `get_device_name(0) == 'NVIDIA B300'`, `total_memory == 288 GiB`,
`compute_capability == 12.0`, `torch.cuda.init()` succeeds.

**The SIGSEGV (core 4023312, 09:35:55 PST), diagnosed and fixed.**  The child died in
`fwd_cuCtxGetCurrent (libcuda.so.1+0x27750)` called from cublas gemm.  The generated
trampoline did `movq fwd_cuCtxGetCurrent@GOTPCREL(%rip), %rax; testq %rax,%rax;
jmp *%rax`: GOTPCREL yields the ADDRESS OF THE POINTER VARIABLE, so the jump landed in
`.data` (SEGV_ACCERR) instead of jumping through the function pointer.  Direct
enumeration never hit it because those functions are overridden; the first FORWARDED
call (cublas's `cuCtxGetCurrent`) did.  Fixed by adding `movq (%rax), %rax` in
`gen_forwarders.py`; every forwarded call now jumps through the slot.

**Measured after both layers, one invocation (`probe_vdev.py`):**

    VDEV OK: device_count=8 name=NVIDIA B300 total=288.0 GiB
             peak_rss=920 MiB mem_available=12435 MiB matmul_err=4.01e-05
    checks: all 10 true (device_count/cudart_count/nvml_count, names, NVML name+total
            match, 288 GiB identity, capability 12.0 real, init, matmul on the GPU)

**D1** is `specs/b300_8x8.json`, generated by `make_b300_spec.py` from the PI's sourced
table: 8 nodes x 8 B300, 288 GiB HBM3e, 8 TB/s, 160 SMs, 1.8 TB/s NVLink, 8 x
ConnectX-8 800 Gb/s, sources per field.  NVLink/NIC latency and SM clock are marked
UNVERIFIED and are not used in an assertion.

**Error budget of this increment:** one real RTX 5090 aliased as eight B300s; the
identity (name, 288 GiB, PCI ids, SM count, PCIe gen) is modelled, arithmetic is real,
and free bytes/capability are forwarded.  No NVLink/IB fabric, no per-node sysfs yet;
those are D2 and 7q3w.3/.5.  The `total_memory` vs `cudaMemGetInfo` split (288 GiB
identity vs real free) is deliberate and is what keeps allocations from over-committing.

## 7. D2 first line: eight node processes, 64 disjoint bus ids

`probe_nodes.py` launches eight processes concurrently, each with its own
`VDEV_NODE=n`, and asks both NVML and cudart for the PCI bus ids.  D1 now carries
`pci_bus_base` (0x0b) and `pci_bus_node_stride` (0x20); `gen_spec.py` emits both
into `b300_spec.h`, and no shim contains the stride as a literal.  Node n, local
device i reports bus `0x0b + 0x20*n + i`, so node n's eight ids are contiguous and
disjoint from every other node's.

Measured (one invocation, `python3 experiments/gpu-sim-7q3w.4/probe_nodes.py`):

    NODES OK: 8 processes x 8 B300, 64 distinct bus ids (0x0b+n*0x20)
              total_peak_rss=4666 MiB min_mem_available=11158 MiB
    8/8 checks: eight_node_processes, every_node_sees_8, every_node_names_b300,
                nvml_and_cudart_agree, 64_distinct_pci_bus_ids,
                bus_set_is_d1_exact, nodes_have_distinct_first_bus, node_n_first_bus

**The measurement trap this found:** the child first loaded
`ctypes.CDLL("libcudart.so.13")` directly and `cudaDeviceGetPCIBusId` returned the
REAL runtime's single device (bus 1, then errors for 1..7).  A direct handle to the
real soname binds the real library's own internal symbol, so the LD_PRELOAD
override is bypassed.  `ctypes.CDLL(None)` (global scope) resolves to the
preloaded shim, which is the layer torch's imports actually bind; with that
correction NVML and cudart agree on all 64 ids.  This is the same class of trap as
the `GOTPCREL` crash: a probe can silently measure the real library and still
print a plausible number.

**Remaining D2 gap:** this is the per-node GPU/NVML identity.  The per-node
sysfs/NIC view (a node sees only its own ConnectX) is not in this increment; the
7q3w.3 `sysfs_spoof.c` is the transport's one-path rewrite, not a per-node tree.
