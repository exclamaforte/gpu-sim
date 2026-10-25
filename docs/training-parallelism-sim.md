# Training-run parallelism simulator — current design

Written 2026-10-23. Describes what exists and how to run it. Supersedes an
earlier draft of this file that proposed an analytic FLOP-counting model; the
project's premise is the opposite of that, and the draft was wrong.

## 0. Where the project is right now (2026-10-24)

Read this first if you are picking the work up from disk rather than from a briefing.

| piece | state |
|---|---|
| axes DP / TP / PP / ZeRO1-3 | all implemented; all train correctly (verified against references, not against each other) |
| context parallel (CP) | implemented: sequence split into contiguous blocks, K/V exchanged over a p2p ring, cp groups. gpu-sim-5o4.3 (2d11df0). Gate: cp=2 forward maxerr 1.9e-8 vs cp=1. train.py still refuses cp>1 combined with pp>1 or zero>1 (unverified pairs). |
| collective cost | ANALYTICAL: n-1 hops of S/n per ring phase, latency vs bandwidth chosen by whether a hop outlasts link latency (`epic_cluster/nccl_cost.py`, mirrored in `src/fakegpu_nccl.c`). Legacy table behind GPUSIM_NCCL_LEGACY_TABLE=1. |
| 5D search | `epic_cluster/parallelism_search.py` ranks (tp,pp,cp,dp,zero) at fixed N by predicted step time: compute+comm SUM (no overlap), with max-overlap reported separately. Those two policies disagree completely -- see below -- so the ranking is conditional. |
| Perfetto | `tools/torch_parallel/merge_profile.py` merges torch.profiler kernels/Python with the interposer's collectives; three rows per rank. |
| calibration | **counts** validated on hardware (gpu-sim-5o4.7): 16 tp all-reduces and 24 cp K/V sends per rank per step match the plan exactly. Byte SIZES are not validated: the harness's tp all-reduce carries a 3d/tp qkv tensor while the cost model assumes Megatron's full-d row-parallel payload, so the model's byte magnitudes are Megatron-form and modelled-only. Absolute durations modelled (gpu-sim-wv2, blocked: needs 2-4 real GPUs). |
| attention FLOPs | gpu-sim-ooj added attention (seq^2) fwd+bwd, 12*L*S^2*d per sequence. **The delivered scaling was wrong and its headline claim is WITHDRAWN (PI, gpu-sim-i95):** it divided by cp*tp but not pp*dp, while the weight term beside it divides by tp*pp*cp*dp, so compute stopped being split-invariant and tp/cp configs got a spurious discount -- which is where the 'cp=4 wins at S>=8192' result came from. Scaled like the weight term, 12*L*S^2*d/(tp*pp*cp*dp) = (12*L*S*d)*S/N, the term is split-invariant, raises every config's compute by ~27% at S=8192 and ~107% at S=32768, and the winner stays pure FSDP (verified by direct re-run: 73.06 ms at S=8192, 428.05 ms at S=32768; compute floor 64.97 / 419.96 ms; all 24 pp=1 rows tie at the max-overlap floor). Crossover where attention equals the weight FLOPs is S ~= P/(2*L*d) = 30.6k for 8B/d4096. Conservation test (compute identical for all configs with equal N) tracked in gpu-sim-i95. |
| rank layout (read before touching groups) | rank = ((pp_idx*tp + tp_rank)*cp + cp_rank)*dp + dp_rank. TP groups are stride cp*dp, PP neighbours keep (tp_rank, cp_rank, dp_rank) fixed. gpu-sim-x7h found tp groups built with stride 1 and no cp term, which silently disabled tp all-reduce on every rank with cp_rank>0; fixed in 4e8d65c. `tools/torch_parallel/validate_trace.py` now refuses a run whose ranks have unequal collective counts (stage-aware, and invalid rows yield no step time). |

THE RANKING IS A NO-OVERLAP RANKING, and that is the open question, not a footnote.
flops_per_gpu = (6*P + 12*L*seq*d)*seq/(tp*pp*cp*dp) = (6*P + 12*L*seq*d)*seq/N is
INVARIANT across the split at fixed N: the attention term changes the MAGNITUDE (about +27%
at seq=8192 and +107% at 32768, crossover at seq ~= P/(2*L*d) = 30.6k for 8B/d4096), never
the invariance. So every pp=1 row reports the same compute -- 51.63 ms weight-only, 64.97 ms
with attention at seq=8192 -- and the ORDER is decided entirely by the comm terms, not by
the dominant column, which reads compute everywhere. With the corrected terms (TP 4
all-reduces per layer per step; CP 2*(cp-1) K/V sends per layer) the winner is pure FSDP
(tp=1 pp=1 cp=1 dp=8 zero=3, 59.71 ms weight-only, 73.06 ms with attention on the spec path and
73.05 ms on the legacy single-fabric path, the same 0.013 ms path delta as 5o4.5 reported), with tp=2 cp=4
second at 64.23 ms weight-only. But ALL 24 feasible pp=1 configurations have identical
max-overlap time (64.97 ms with attention; 22 of 24 at 51.63 ms weight-only, where two rows'
comm exceeded the floor) while their SUM spans 73.06-129.62 ms with attention (59.71-98.80
ms weight-only): under full overlap the
model cannot separate them at all. Numbers in this section are weight-only unless labelled;
gpu-sim-i95 regenerates every table with the attention term. gpu-sim-5o4.5 quantified the
overlap question on the merged search: the min-comm config (pure FSDP) is the unique winner
for overlap fraction f < 0.25,
then TIES with a growing set (2 at f=0.25, 7 at f=0.55, 22 at f=1) at the 51.626 ms weight-only
compute floor; no other config is ever strictly better and there are no rank swaps. The same
result holds on the legacy single-fabric path and the spec/topology path (59.708 vs
59.721 ms), so it does not depend on which pricing path produced the table. Absolute step
times remain modelled -- one 5090 cannot hold Llama-3-8B at any decomposition (the search's
own memory column says 68.5 GB for the tp=2 cp=4 row and 32.7 GB for the FSDP row with the
completed memory model, gpu-sim-6y1). MEMORY FEASIBILITY IS MODELLED, not measured: the
model now carries params, grads, fp32 master, optimizer, per-tensor activations (FFN at
d_ff), logits and CP buffers, but its activation term is an UNVALIDATED upper bound (torch
frees buffers during backward), so a phase-map cell within ~15% of the 80 GB limit may flip
when that term is calibrated. Everything else in the feasibility column is arithmetic on
bytes that do not depend on the device. What IS validated on this host: the
per-rank collective COUNTS (16 tp all-reduces and 24 cp K/V sends per rank per step,
gpu-sim-5o4.7). What is NOT: the byte SIZES (harness reduces a 3d/tp qkv tensor, the model
assumes Megatron's full-d payload -- gpu-sim-oez) and wall time, ever.

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

Honest status per axis -- traffic is traced for all four, but only DP has been
shown to train correctly (see §6 for what each gate was):

| axis | collective traffic | trains correctly |
|---|---|---|
| DP | yes | yes (world2 == world1 to 1.13e-07) |
| ZeRO 1/2/3 | yes | not yet -- payload work in progress |
| TP | yes | not yet -- payload work in progress |
| PP | yes (+ p2p) | not yet -- payload work in progress |

Reading the left column as if it were the right one is the single easiest mistake
to make with this harness, which is why they are separate columns.

| Piece | Where | State |
|---|---|---|
| CUDA runtime + driver interposer | `src/`, `shim/cuda_runtime.h` | 89 exported `cuda*`/`cu*` symbols; random fill; fixed, bandwidth, table-driven or synthetic latency |
| NCCL interposer | `src/fakegpu_nccl.c`, `shim/nccl.h` | 68 symbols — now the **full set torch references** (see §4) |
| NCCL-only build | `libfakenccl.so` + `shim/exports_nccl.map` | exports only `nccl*`, so torch computes on the real GPU with fake collectives |
| Trace emission | `fg_nccl_trace`, `GPUSIM_TRACE=<path>` | JSONL, one line per collective per rank |
| Fabric cost model | `epic_cluster/link_model.py` | predicts collective duration from a declared topology; measured for n=2, refuses n≥3 without explicit `pieces_per_phase` |
| Torch harness | `tools/torch_parallel/` | runs a real transformer; DP, TP and PP axes; sweeps and summarises |
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

**Every subgroup shared one comm identity.** `ncclGetUniqueId2` returned a
constant, so torch's comm cache aliased distinct groups: with pp2/world4 the group
`[2,3]` was never `ncclCommInitRank`'d and its traffic was silently attributed to
another group's communicator. The id must be identical across the ranks of a group
and distinct per group — a per-process call counter satisfies both. Related: the
global rank was read from an env var the harness never set, so every process
reported rank 0; traces now carry both `rank` (comm-local) and `grank` (world).

**The NCCL datatype enum was permuted against real NCCL.** `shim/nccl.h` had
Float32=6, Float64=7, Float16=8; real NCCL has Float16=6, Float32=7, Float64=8.
Every float32 volume was therefore reported at 8 bytes/element — exactly 2× too
large, across an entire measurement campaign. It survived because it was
self-consistent: every ratio, ordering and conclusion held while every absolute
number was wrong. A regression test now pins the constants, mutation-checked
against the original defect.

**`ncclReduceScatter` took its count by pointer.** Real NCCL passes `recvcount` by
value; our shim declared it `const size_t *`, so torch's integer count was
dereferenced as an address — a guaranteed SIGSEGV on any FSDP/ZeRO run. This is
worse than a missing symbol: a missing one fails at load with a clear message, while
a calling-convention mismatch links silently and faults later somewhere that looks
unrelated. Found from a coredump backtrace, not from reading the header.

That makes five defects of one family — a constant `uniqueId`, a permuted datatype
enum, untraced p2p, a wrong parameter type, and the 462-symbol driver dead end.
**None was found by reading the declaration. Every one was found by probing the
call.** For any further work on the shim surface, that is the rule.

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

### All three axes, same world size

World=4, NVLink, 5 steps with 2 discarded as warmup. Re-run from a clean worktree
and reproduced bit-identically:

```
config  step_ms  coll  coll_MB  p2p  p2p_MB  comm_sim_ms  params  local
DP4        16.1    75     43.0    0     0.0         0.57  5242880  5242880
TP4         9.7    33    134.2    0     0.0         0.81  5242880  5242880
PP2        13.5    37     22.0    8    16.8         0.35  5242880  2621440
PP4         3.9     1      0.0    8    16.8         0.03  5242880  1967104
```

Read `local` before `step_ms`: DP and TP keep the whole model on every rank, PP
does not. PP4 is fast mostly because each rank owns a quarter of the layers, not
because pipeline parallelism is efficient -- the column exists to stop that
misreading.

The shape of each trade-off is visible and explicable:

- **DP** issues many small collectives (one per parameter, un-bucketed here).
- **TP** issues fewer, much larger ones (2 per layer per direction).
- **PP** issues few collectives and less total bytes, plus point-to-point traffic
  between stages.

**The PP bubble does not appear in wall time here**, and that is a property of the
harness, not a result: a simulated p2p hop costs microseconds while compute is
real, so stages barely wait. Reporting a bubble would require compute to be
simulated too.

### Data parallel vs tensor parallel (world=2)

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
measured one. The default remains REFUSE.

### Pipeline parallel, same world size

```
config     world  tp pp dp  step_ms  coll_msgs  coll_MB  p2p_msgs  p2p_MB  comm_sim_ms
DP             4   1  1  4     16.1         75     43.0         0     0.0         0.57
TP             4   4  1  1      9.7         33    134.2         0     0.0         0.81
PP2            4   1  2  2      7.5         37     22.0         8    16.8         0.35
PP4            4   1  4  1      4.1          1      0.0         8    16.8         0.03
```

Measured on the same box and model (5.2M parameters, 5 steps, 2 discarded,
nvlink, `GPUSIM_ALLOW_EXTRAPOLATE=1`). Counts are per grank 0 over the 2
measured steps and decompose exactly: PP2 collectives 37 = 18 stage params x 2
+ barrier, p2p 8 = 4 per step x 2 at 2.1 MB each (4x512x256 float32 =
numel x element_size); PP4 collectives 1 = barrier only. `step_ms` absolutes
move run to run on the shared box; the traffic counts are bit-identical
across runs, so traffic is the signal and step time is context.

The shape is explicable: PP replaces TP's few huge allreduces with smaller,
more numerous messages (PP2: 45 total messages, 38.8 MB vs TP's 33 messages,
134.2 MB), and pure pipeline (PP4, dp=1) drops to 9 messages and 16.8 MB.
PP2 carries MORE collectives than TP4 (37 vs 33) because at pp=2 each stage
still data-parallel-syncs its half-model across dp=2 -- the "fewer
collectives" premise holds for the pipeline axis proper (PP4), not when DP
co-traffic is counted alongside it.

The bubble is ABSENT from wall time, and that is stated plainly rather than
fitted: the interposer's send/recv return instantly (timing only, no payload),
so no stage ever actually blocks on another and there is no fill/drain idle
to measure. Per-rank means after the backward fix are within ~10% (stage 0:
11.5/13.5 ms, stage 1: 12.4/12.7 ms); the spread is residual compute
imbalance, not pipeline idle. The modelled p2p cost (predicted_s, p2p-link
regime) totals 0.35/0.03 ms against milliseconds of compute -- even modelled,
there is no meaningful bubble at this model size. A wall-clock bubble would
need a transport that actually blocks, which is the data-correct-collectives
bead, not this one.

Two defects found and fixed along the way, both load-bearing for the numbers
above. (1) The last stage's backward used `autograd.grad(loss, stage_output)`,
which leaves every param grad None: the DP sync silently skipped a whole
stage and the stage ran 3x fast. Now `retain_grad` + full `backward`, verified
by per-rank times converging. (2) Subgroup traffic was mislabeled (local rank)
and partly missing (constant ncclUniqueId aliased distinct groups, so dp
group [2,3] was never even initialized) -- fixed on master (gpu-sim-vtu):
distinct uniqueIds per group and both `rank` (comm-local) and `grank` (world)
on every trace line; the sweep aggregates on `grank`.

### ZeRO, same world size

```
config     world  tp pp dp zero  step_ms  coll_msgs  coll_MB  comm_sim_ms  peak_MB  state_MB
DP/ZeRO-1      4   1  1  4    1     14.6        149     86.1         0.93    627.2      52.4
ZeRO-2         4   1  1  4    2     16.0         77     53.8         0.41    675.6      36.7
ZeRO-3/FSDP    4   1  1  4    3     11.6          5     21.5         0.09    650.6      21.0
```

Same box, model, steps, fabric. Counts per grank 0 over the 2 measured steps;
peak_MB is the measured `torch.cuda.max_memory_allocated` windowed like the
trace (reset at measure_start); state_MB is the analytic sharded-state model
(P + P + 2P sharded by stage; P = 21.0 MB fp32). Decomposition is exact:
ZeRO-1 = 75 DP + 74 param-gathers + barrier (86.1 = 43.0 + 43.0 MB);
ZeRO-2 = 2 reduce_scatter (5.4 MB shards) + 74 gathers + barrier;
ZeRO-3 = 2 allgather + 2 reduce_scatter + barrier (5.4 MB each).
P_actual moved is 21.5 MB, not 21.0: `param_count()` excludes pos/LN.

The deliverable trade-off holds: per-rank traffic bytes fall with N
(86.1 -> 53.8 -> 21.5) while step time rises 1 -> 2 (14.6 -> 16.0, the extra
74 gathers). ZeRO-3 breaks the rising-time half: 11.6 ms, fastest of the
three, because 4 collective launches replace 149 and the optimizer steps a
shard. Stated, not fitted.

Memory is the honest divergence this axis exists to show: measured peak
barely moves (627/676/651 MB -- activations dominate, no ZeRO stage shards
them) while the analytic state falls 52.4 -> 36.7 -> 21.0 MB. ZeRO-2's peak
is HIGHEST (+48 MB over ZeRO-1): the hand-written path allocates flat grad
(21 MB), shard (5 MB), and per-param gather scratch every step. That scratch
is the price of the timing-event approach and it is measured, not modelled.

How each stage is built: ZeRO-1 = DP allreduce + post-step all_gather into
discarded scratch (the interposer moves no payload, so copying back would
corrupt good params with uninitialized memory). ZeRO-2 = one flattened
reduce_scatter + the same post-step gather; the local update still uses full
grads (timing events, not arithmetic). ZeRO-3 = FSDP over the dp group.
FSDP was predicted blocked on payload; it is not -- the only blocker was our
own ReduceScatter signature bug (recvcount by pointer vs by value, SIGSEGV in
backward, fixed on master with a parse test pinning it). tp/pp x zero
combinations are refused explicitly until verified.

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
- **PP.** Done (gpu-sim-8y7.1): stage split of TinyLM layers across the pp axis,
  GPipe schedule (`--micro-batches`), blocking send/recv between adjacent stages,
  `--pp`/`--micro-batches` in train.py and sweep.py with rank decomposition
  rank = pp_idx*(tp*dp) + dp_idx*tp + tp_idx. Same timing-only bargain as the rest:
  stage outputs are garbage, loss curves are not meaningful, traffic/timing are real.
- **Memory, and one rule for reading it.** ZeRO's modelled sharded-state saving is
  52.4 -> 36.7 -> 21.0 MB, and with payload really moving the measured peak is
  29.0 MB lower for ZeRO-3 than ZeRO-1 -- within 8% of the model. Two caveats,
  both earned: absolute peak numbers are only comparable between runs with the
  SAME BATCHING CONVENTION (the `--dp-mean` sharding convention alone moves peak
  from 627 MB to 272 MB at world=4, entirely independent of transport mode), and
  the transport itself has zero measurable memory effect.
- **ZeRO.** Done (gpu-sim-8y7.3): `--zero 1/2/3` on the dp axis. 1 = DP +
  post-step all_gather; 2 = flattened reduce_scatter + post-step gather;
  3 = FSDP over the dp group. Measured peak (`max_memory_allocated`,
  windowed) reported alongside the analytic sharded-state model; tp/pp
  combinations refused until verified.
- **Data-correct collectives.** all_reduce only, done (gpu-sim-n5h): real sums
  through /dev/shm (d2h, barrier, host reduce, h2d), gated behind
  FAKEGPU_NCCL_SIMULATE_ONLY=0; default stays simulate-only and reproduces
  the committed traffic record byte-exact. float32/bf16, sum only; all_gather
  and the rest still move nothing.
- **Training correctness under DP.** Done (gpu-sim-tbp): `--dp-mean` selects true
  DDP (disjoint rank shards of one seeded batch, explicit 1/dp average), and
  world=2 agrees with world=1 to worst 1.13e-07 across 12 steps; 50 steps fall
  8.4806 -> 8.4276 with zero up-steps. The convention is not a detail: with the
  same batch on every rank the summed gradient is dp x the single-rank gradient,
  so "matching losses" would prove only that both sides shared one wrong
  convention. Transport-on without `--dp-mean` refuses rather than defaulting.
- **Single-process multi-device enumeration.** Blocked on NVML interposition.

## 7. Non-goals

- **Convergence claims about real hardware.** The harness can now train correctly
  under data parallelism, but it does so on one GPU with a simulated fabric, so its
  loss curve is evidence about the SOFTWARE configuration, not about a cluster.
  (This non-goal used to read "the harness moves no data between ranks" — that
  stopped being true when the transport landed, and leaving it would have
  understated the system as badly as claiming more than it does.)
- Bit-accurate step times without calibration.
- Re-implementing the model's arithmetic at a higher level. If a number cannot be
  observed at the library boundary, this project does not report it.
