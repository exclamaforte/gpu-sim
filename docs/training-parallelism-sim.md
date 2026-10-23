# Training-run parallelism simulator — current design

Written 2026-10-23. Describes what exists and how to run it. Supersedes an
earlier draft of this file that proposed an analytic FLOP-counting model; the
project's premise is the opposite of that, and the draft was wrong.

## 1. Premise

**Intercept the libraries, don't re-implement the model.**

A training program written against torch / NCCL runs unmodified. The CUDA and
NCCL symbols it calls are resolved to our interposer instead of the real ones, so
timing and traffic decisions are made below the Python line. The same program can
then be run against different parallelism configurations and different fabrics
without being rewritten, and the trace that comes out is a record of what the
program *actually asked the hardware to do* — not a spreadsheet's opinion about
it.

North star: run a real torch model under different parallelism configurations
(DP now; TP/PP later), see the collective traffic and the time it costs, and use
that to pick a configuration before committing GPUs to one.

Accuracy is explicitly not the first bar. With no calibration data, per-kernel
times are synthetic (random within a declared range) and collectives are timed by
the declared fabric model. What must be right first is the **structure**: which
collectives are issued, with what sizes, in what order, by which rank.

## 2. What exists

| Piece | Where | State |
|---|---|---|
| CUDA runtime + driver interposer | `src/`, `shim/cuda_runtime.h` | 89 exported `cuda*`/`cu*` symbols; random fill; fixed, bandwidth, table-driven or synthetic latency |
| NCCL interposer | `src/fakegpu_nccl.c`, `shim/nccl.h` | 68 symbols — now the **full set torch references** (see §4) |
| NCCL-only build | `libfakenccl.so` + `shim/exports_nccl.map` | exports only `nccl*`, so torch computes on the real GPU with fake collectives |
| Trace emission | `fg_nccl_trace`, `GPUSIM_TRACE=<path>` | JSONL, one line per collective per rank |
| Fabric cost model | `epic_cluster/link_model.py` | predicts collective duration from a declared topology; measured for n=2, refuses n≥3 without explicit `pieces_per_phase` |
| Torch harness | `tools/torch_parallel/` | runs a real transformer, sweeps world sizes, summarises |
| Rack planner (analytic) | `cy0_l1..l4.py` | pre-existing FLOP/roofline planner; **separate track**, not used by the harness |

## 3. How a run works

```
torchrun --nproc_per_node=N  tools/torch_parallel/train.py
   └─ torch program (unchanged): TinyLM, AdamW, backward
        ├─ CUDA calls   → real GPU (compute is real, on the physical card)
        └─ NCCL calls   → libfakenccl.so
                            ├─ simulated duration (fabric model, or fallback)
                            └─ one JSONL line per collective: rank, op, dtype,
                               count, bytes, timestamp, predicted_s
```

Two facts that make this work, both learned the hard way (see §4):

- **Ranks share the one real GPU.** The multi-device fan-out is simulated. Every
  rank calls `torch.cuda.set_device(0)`; asking for device==rank is what
  "invalid device ordinal" means on a one-GPU box.
- **Collectives simulate timing, not data.** Buffers are left untouched
  (`FAKEGPU_NCCL_NO_FILL=1`), because writing into a real device pointer from the
  host segfaults, and because only the timing is being simulated. Consequence:
  after `all_reduce` the gradient is not the true sum across ranks. Traffic and
  timing are real; the arithmetic is not.

## 4. Findings that changed the design

**A single `LD_PRELOAD` of `libfakegpu.so` cannot drive torch.** It replaces
`cuda*` and `cu*`, but torch's real `libcudart` resolves the driver through
`dlopen("libcuda.so.1")` + `dlsym`, which preload does not interpose. The process
ends up with fake handles reaching the real driver and dies inside
`cuEventRecordWithFlags`. Standing in for the driver instead means implementing
**462** `cu*` symbols that cudart looks up by name — measured, not estimated.

**So the interposer is split.** `libfakenccl.so` exports `nccl*` only. CUDA stays
real, NCCL is simulated. That is the split that runs torch today.

**torch references 51 `nccl*` symbols; the interposer exported 38.** The missing
ones included `ncclCommInitRankConfig` — the call `ProcessGroupNCCL` actually
makes — so torch silently fell through to real NCCL and failed in
`ncclSocketInit`. The full set is now covered (config init, unique ids, user
rank, `ncclMemAlloc/Free`, group-simulate-end, pre-mul-sum, the `Bcast`/`AlltoAll`
/`Gather` spellings, and benign success stubs for the window/device-comm/signal
extensions a plain run never exercises).

**Device count comes from NVML, not CUDA.** `torch.cuda.device_count()` consults
`nvmlDeviceGetCount_v2` on the real driver. Interposing CUDA alone will not make
torch see more devices. Relevant only for single-process multi-device programs;
the harness sidesteps it by using one process per rank.

**`torch.nn.parallel.DistributedDataParallel` cannot be used as-is.** DDP
cross-checks parameter counts with a real all_gather, so it needs collectives that
move payload. The harness does the gradient all-reduce explicitly instead — the
same traffic DDP would produce, un-bucketed.

## 5. Running it

```bash
make libfakenccl.so                      # once
python3 -m tools.torch_parallel.sweep --world-sizes 1 2 --steps 5
python3 -m tools.torch_parallel.sweep --world-sizes 2 --fabric pcie
python3 -m tools.torch_parallel.sweep --world-sizes 1 2 --delay-ms 0.5 2.0
```

The sweep emits a topology blob for each world size
(`tools/torch_parallel/topology.py`) and points `GPUSIM_TOPOLOGY` at it, so
collectives are timed by the declared fabric. Without it they take the fallback
path and carry `predicted_s: null` -- bytes but no cost.

Measured on this box (RTX 5090, 5.2M-parameter model, 5 steps, 2 discarded as
warmup):

```
fabric  world   step_ms  coll_msgs   coll_MB  comm_sim_ms
nvlink      1      11.4          0       0.0          0.0
nvlink      2      11.5         75      86.1          0.68
pcie        2      12.1         75      86.1          7.04
```

Three things that table is good for, none of which needed a second GPU:

- doubling data-parallel width adds 75 collectives and 86 MB per rank;
- the same configuration on a slower declared fabric costs 10x more simulated
  communication (0.68 -> 7.04 ms), so fabric and parallelism can be co-varied;
- `step_ms` barely moves because the simulated collective time is small next to
  this model's compute -- which is itself the answer to "is this model
  communication-bound?": at 5.2M parameters and 8x512 tokens, no.

### Data parallel vs tensor parallel, same world size

```
config     world  tp  dp  step_ms  coll_msgs  coll_MB  comm_sim_ms
DP             2   1   2      9.1         75     43.0         0.62
TP             2   2   1      9.9         33    134.2         0.28
```

These are CORRECTED volumes. Every figure published before 2026-10-23 was
**exactly 2x too large** for float32: shim/nccl.h defined Float32=6, Float64=7,
Float16=8, while real NCCL defines Float16=6, Float32=7, Float64=8. torch passes
the constants it compiled against, so float32 was decoded as float64. The error was
invisible because it was self-consistent -- every comparison, every ratio, every
"TP moves more bytes than DP" conclusion was unaffected, which is exactly why it
survived so long. A 1000-element probe caught it: 4000 B of float32 traced as
8000 B. The enum now matches the ABI.

The shape of the trade-off is visible and explicable: **TP issues fewer, much
larger collectives; DP issues many small ones.** That is the textbook tension, and
it is now a measurable property of a config rather than an assertion.

One caveat that keeps this from being a like-for-like byte comparison: DP here is
UNBUCKETED (one all-reduce per parameter, 37 of them per step), whereas DDP would
fuse them into a few large ones. Un-bucketed is a deliberate upper bound on
message COUNT and easier to audit from the trace; it is not what a tuned DDP run
would emit.

TP collectives are 8 MB, beyond the measured phase-payload table, so they are
timed only when `GPUSIM_ALLOW_EXTRAPOLATE=1` is set; the trace then records
`regime: "extrapolated"` so an extrapolated cost can never be mistaken for a
measured one. The default remains REFUSE.

### Warmup

WARMUP IS DISCARDED, and that matters: the first step measured 556 ms against
11 ms for steady state (CUDA context, cuBLAS init, kernel compilation). Averaging
it in would make a sweep compare initialization cost instead of the parallel
configuration. `coll_msgs` is filtered to the same window as `step_ms`
(`measure_start_ns`) so the two describe the same steps.

## 6. What is not built yet

- **Fabric-backed timing on the harness path.** Without `GPUSIM_TOPO`, collectives
  take the fallback duration and are traced with `predicted_s: null`. Wiring
  `link_model.predict()` into that path is the next step, and it is what turns
  "we issued 75 collectives" into "they cost X ms".
- **Point-to-point timing.** `send`/`recv` cost `S/min_bw(path) + sum(latency)`
  from the declared link parameters — first principles, not the fitted ring table,
  which is why p2p is modelled while large ring collectives are still refused.
  Verified: 256 B on NVLink 1.0003 us, on PCIe 8.0102 us. An unconnected pair
  refuses; with no topology the duration stays null.
- **PP.** TP exists (Megatron-style column/row sharding, `_ReduceSum` makes the
  backward reduction visible so TP traffic is not understated by half). PP does
  not: it needs stage split and per-stage send/recv; the interposer already traces
  `send`/`recv`, so the work is a scheduler, not a C change.
- **Data-correct collectives.** Would need a transport between ranks (shared
  memory or sockets in the interposer). Not required for timing comparison;
  required before any convergence claim.
- **Single-process multi-device enumeration.** Blocked on NVML interposition.

## 7. Non-goals

- Convergence. The harness moves no data between ranks.
- Bit-accurate step times without calibration.
- Re-implementing the model's arithmetic at a higher level. If a number cannot be
  observed at the library boundary, this project does not report it.
