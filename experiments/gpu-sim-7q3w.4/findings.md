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
| child: torch + 12.6M-param transformer + AdamW step | **1148 MiB** |
| 8 nodes x 1148 MiB | 9184 MiB |

**The old 970 MiB row was the 3.16M model, not the 12M one.**  The model was
labelled ~12M but `sum(p.numel())` = 3,159,040; the corrected d_model=512 /
ff=2048 / 4-layer shape is 12,609,536 params and measures 1148 MiB.  So the
bead's estimate (~0.7 GiB/node) was low by ~64% once the real model and
optimizer are resident: the design needs **9.2 GiB**, which fits neither sample
(8991 MiB is 193 MiB short even before a margin).  The consequence is a
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

## 8. A1's first sentence: torch's own NVML load and the `_v2` trap

The PI read the requirement out of this venv's source: torch opens
`CDLL("libnvidia-ml.so.1")` itself at `torch/cuda/__init__.py:950` and `:1004`
(pynvml is not installed), and the COMPLETE set of entry points it calls is four:
`nvmlInit`, `nvmlDeviceGetCount_v2`, `nvmlDeviceGetHandleByIndex_v2`,
`nvmlDeviceGetUUID`.  The `_v2` suffixes are mandatory: a shim exporting only the
unsuffixed names raises `AttributeError`, torch catches it, returns -1, and falls
back to `_cuda_getDeviceCount` -- which looks like "NVML was never consulted" when
it was consulted and refused.  Both spellings are exported in `shim/nvml.c`.

The load is the same class of trap as the cudart one in sec. 7.  A path-only
`LD_LIBRARY_PATH` entry loses if anything pulls the vendor library in first.
The robust form, now used by both probes: the file is named
`libnvidia-ml.so.1`, built with `-Wl,-soname,libnvidia-ml.so.1`, and
`LD_PRELOAD`ed, so torch's `CDLL("libnvidia-ml.so.1")` matches OUR already-loaded
object by soname.  `/proc/self/maps` in the child shows both
`.lab/tmp/vdev-build/libnvidia-ml.so.1` (the preload, which torch binds) and
`/usr/lib/libnvidia-ml.so.615.78.08` (the real library, loaded RTLD_LOCAL by the
shim's constructor by ABSOLUTE path for the 400+ forwarded entry points).  The
absolute path is what avoids the self-dlopen recursion the driver shim hit.

UUIDs: D1 now carries `uuid_prefix: "GPU-VDEV"`; `gen_spec.py` emits
`B300_UUID_PREFIX`, and NVML, cudart and the driver all produce the SAME
`GPU-VDEV-N<node>-D<device>` string (16 bytes in `cudaDeviceProp.uuid`, 15 chars
plus NUL).  With `CUDA_VISIBLE_DEVICES` unset torch's `_device_count_nvml()`
uses `nvmlDeviceGetUUID`; 8 distinct node-local UUIDs is what keeps the count at
8 instead of -1.

Measured in one child with `VDEV_NODE=3` (`probe_vdev.py --node 3`):

    VDEV OK: node=3 device_count=8 nvml_private=8 name=NVIDIA B300 total=288.0 GiB
             peak_rss=917 MiB mem_available=13378 MiB matmul_err=4.01e-05
    14/14 checks, including nvml_private_count_is_N, nvml_uuids_distinct,
    nvml_uuids_are_node_local, nvml_preloaded_object_mapped, and the matmul

No shim contains a 288/GiB/8/B300/1.8/800 identity literal: the name, memory,
SM count, PCI base/stride and UUID prefix all come from the generated
`b300_spec.h`.  The real card's capability (12.0) is forwarded, as A1 requires.

## 9. Deliverable increment 1: eight concurrent node processes and the preflight

The capacity probe's "~12M params" label was wrong: its d_model=256, 4-layer
model is 3,159,040 params, and the 936 MiB figure belonged to that shape.  The
corrected d_model=512 / ff=2048 / 4-layer model is 12,609,536 params and measures
1148 MiB/node with AdamW after one CPU step.  The preflight budget is therefore
8*1148 + 1024 margin = 10208 MiB, not 7488 + 1024.

The preflight refused when it should: at MemAvailable 9889 MiB it printed REFUSE
and launched nothing.  At MemAvailable 11570 MiB it ran all eight children
concurrently, each with its own VDEV_NODE:

    CONCURRENT OK: 8 processes device_count=8 name=NVIDIA B300
                   rss_sum=9128 MiB (budget 9184 MiB)
                   mem_available_start=11570 MiB end=10305 MiB
                   preflight_required=10208 MiB
    per-node RSS MiB: 1141 1133 1154 1127 1137 1147 1140 1149
    each child: params=12609536, device_count=8, name='NVIDIA B300'

**Tool trap:** the detached `run` tool is CPU-only by contract and hides CUDA
(`CUDA_VISIBLE_DEVICES=""`), so the first background attempt died inside AdamW's
`_accelerator_graph_capture_health_check()` with `cudaErrorNoDevice` even though
the shims were fine.  The probe must run in the agent shell with the real device
visible; use the run tool only for waiting, then run the probe manually.

## 10. Deliverable increment 2: torch.distributed, world_size 8, agreement

`probe_distributed.py` starts eight concurrent ranks, one per node
(`VDEV_NODE=0..7`), joins a gloo process group over loopback, and prints each
rank's node index plus the eight PCI bus ids it sees.  The group then does a
real `all_reduce` of `rank+1`; SUM(1..8)=36.0 must appear on every rank.

Measured (one invocation, MemAvailable 8656 MiB start / 7902 MiB end):

    DIST OK: 8 ranks x 8 B300, 64 distinct bus ids (0x0b+n*0x20),
             all_reduce=36.0 on every rank, rss_sum=4741 MiB,
             mem_available_start=8656 MiB end=7902 MiB
             preflight_required=5824 MiB
    per-rank RSS MiB: 594 593 591 594 589 593 593 594
    each rank: device_count=8, name='NVIDIA B300', 8 bus ids
    node 0: 0x0b..0x12, node 1: 0x2b..0x32, ..., node 7: 0xeb..0xf2
    all_reduce_values: [36.0 x 8]

Checks: eight_ranks, each_rank_is_its_node, process_group_world_size_8,
every_rank_sees_8, every_rank_names_b300, nvml_and_cudart_agree,
64_distinct_pci_bus_ids, bus_set_is_d1_exact, all_gather_sees_every_node,
all_reduce_agrees -- all true.

The agreement checker is fail-able, and that was measured, not assumed:
`--inject-disagreement` makes rank 3 add 1.0 after the reduction.  The control
run detected `[36, 36, 36, 37, 36, 36, 36, 36]` and printed it verbatim, exit 0
for the control (a disagreement is the result the checker exists for).  A real
disagreement in the non-injected run would print the same values and exit 1.

This is the gloo CPU control plane.  It makes no NCCL data-path claim; that is
7q3w.3/7q3w.5.  Increment (3) (the 12.6M training run) starts only after this
posts.

## 11. Deliverable increment 3: 12.6M transformer, 8 ranks, equal checkpoints

`probe_training.py` builds the 12,609,536-parameter transformer on every rank,
wraps it in `DistributedDataParallel` over gloo, and takes three real AdamW
steps.  Each rank trains on its own rank-sharded tokens, so the equal checkpoint
is a product of the gradient all-reduce, not identical inputs.

Measured (one invocation, MemAvailable 11638 MiB start / 13231 MiB end):

    TRAIN OK: 8 ranks x 12,609,536 params, 3 real AdamW steps,
              checkpoint_sha256=f30a6a97343ded9c45d86f37421435680031cfd64152069e7b605e12e8afc555
              equal on all 8 ranks; rss_sum=9856 MiB
              mem_available_start=11638 MiB end=13231 MiB
              preflight_required=10576 MiB
    rank 0 losses: 0.9999922514, 0.9975503683, 0.9929448366
    every rank printed 3 finite losses (different data -> slightly different
    loss values); every rank: device_count=8, name='NVIDIA B300'

All 8 ranks wrote `rankN.pt`; the reported hash is canonical over state_dict
key order, dtype, shape and CPU tensor bytes (container metadata excluded).  A
separate 12.6M-element gloo all_reduce produced the same SHA-256 on all eight
ranks, so the DDP equality is measured rather than assumed.

The equality checker is fail-able: `--inject-checkpoint-difference` perturbs
one parameter on rank 3 after training.  The control run detected seven copies
of `f30a6a...` and rank 3's `ba0f82e95780d75e85c32bcc06c85b8fb3af99b8cbc580fd63e29dd01141b803`,
and printed the per-rank hashes verbatim.

The training math is CPU float32.  It is a real distributed training run (DDP
gradient all-reduce, rank-sharded data, AdamW); it makes no NCCL data-path
claim.  The GPU identity is A1's, and the NCCL path is 7q3w.3/7q3w.5.


## 12 Deliverable package: A-items, D4, D1 latency, error budget

### 12.1 A-item -> probe map (one line each)

- **A1** -> `probe_vdev.py`; checks `device_count_is_N`, `name_is_D1`,
  `total_mem_is_D1`, `sm_count_is_D1`, `uuid_prefix`, `nvml_private_count`,
  `matmul_err`; run is increment 1 (8 virtual B300 on one node).
- **A2** -> `probe_a2.py`, which drives `probe_nodes.py` twice: the normal run
  must produce `64_distinct_pci_bus_ids` with each child reporting exactly 8
  ids; the `--all-node 0` run pins every child to node 0 and must FAIL
  `64_distinct_pci_bus_ids` (`distinct ids = 8 != 64`), `bus_set_is_d1_exact`
  and `nodes_have_distinct_first_bus`, exiting 1 as expected. Run is increment 0.
- **A3** -> moved to `gpu-sim-s2vc` (NCCL topology XML/IB choice). Not claimed.
- **A4** -> `gpu-sim-7q3w.5` (NCCL numerical path). Increment 2 is a gloo
  control-plane all_reduce, not an A4 result.
- **A5** -> `probe_reference.py`: one 8-way DDP run and one 1-way run with the
  same seed and per-(rank, step) data generators; compare final mean loss over
  the eight shards and the max-abs parameter delta. Tolerance `5e-4` absolute
  (measured max drift 3.295e-4 from float32 all-reduce order; final loss drift
  1.49e-8).
  Run is increment 4; the measured line is `A5 OK` in `output.log`.
- **A6** -> moved to `gpu-sim-ae2v` (fabric timing). Not claimed.
- **A7** -> not separately rerun here. Increment 3's checkpoint hash was
  reproduced by the PI; the 20-run exit distribution remains `7q3w.3`.
- **A8** -> the RAM error budget in section 12.4 and `metrics.json`.

### 12.2 D4 (model/launcher) coverage

D4 is covered by the three probes below; it is not a separate unprobed claim:

- model: `probe_training.py` / `probe_reference.py` -- 4-layer
  TransformerEncoder, d_model 512, ff 2048, **12,609,536 parameters**, AdamW
  lr=1e-3, 3 steps, 8 DDP ranks, rank-sharded data;
- launcher: `run.sh` selects one increment; `capture.py` waits for that
  increment's own preflight floor, records exit code and MemAvailable
  before/after, and writes `output.log`, `metrics.json`, `job.json`;
- identity: `probe_vdev.py` (A1) and `probe_nodes.py`/`probe_a2.py` (A2/D2)
  supply the 8x8 virtual-cluster identity the model runs on.

D1 identity is `probe_vdev.py`; D2 is `probe_nodes.py` + `probe_a2.py`; D3 is
`probe_vdev.py` (real driver/cudart/NVML through the shims) plus
`probe_distributed.py` (one rank per node).

### 12.3 D1 per-link latency: MODELLING CHOICE, not a vendor number

`specs/b300_8x8.json` sets NVLink per-hop `latency_us = 1.0` and NIC per-hop
`latency_us = 5.0`. These are **MODELLING CHOICES** derived from the fabric
calendar in `epic_cluster/nccl_cost.py` (`FABRICS` for `nvlink4_h100` and
`ib_400g`); they are not sourced B300 NVLink or ConnectX-8 figures. The clock
field remains `UNVERIFIED` and `make_b300_spec.py` refuses to invent it. A
reviewer should not read the latency fields as measurements.

### 12.4 RAM error budget (numbers from actual runs)

- preflight floor for the 8-rank training increment: **10576 MiB** =
  8 x 1194 (measured 2-rank DDP RSS) + 1024 MiB margin;
- PI's passing training run: `mem_available_start=10633` MiB -- **57 MiB**
  above the floor; end 13231 MiB;
- this capture's training run: `mem_available_start=13471` / end 14026 MiB
  (the increment-3 quadruple in `output.log`; the earlier `c210e804` run's
  11638/13231 is not the artifact run);
- PI's refusal run: `mem_available_start=10496` MiB < 10576 -- **REFUSE**, no
  rank launched (exit 2).

**Error-budget sentence:** fits on a quiet box with a few hundred MiB of
margin; does not fit on a busy one, and the preflight refuses rather than
OOM-killing. The refusal is fail-able evidence, reproduced in `output.log` by
increment 5 (`--force-refuse-mib 10496`) with the same preflight line.

### 12.5 Identity literals

A grep of `shim/*.c` finds `288` only in comments; `B300`/`8`/capacity/SM
numbers are consumed through the generated `b300_spec.h` macros. The shims do
not hard-code the identity numbers used by the assertions.
