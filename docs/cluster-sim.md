# Direction: arbitrary-cluster simulation on one box

Status: direction set 2026-10-20 by PI (session `lab-pi`). Backlog: epic
`gpu-sim-cl1`. Prior compute-timing track (`gpu-sim-u2o`) is **not** superseded;
this direction composes with it.

## 1. The target, stated so it can fail

Run an unmodified multi-GPU workload (torch / NCCL) against a **declared**
cluster: N devices with per-device models, connected by a declared fabric
(NVLink, PCIe, IB, RoCE, TCP), with link parameters (bandwidth, latency, MTU,
jitter, loss), executed on one machine without root or a kernel module.

The observable behaviour — API results, timing, collective values, failure
modes — must be what the declared cluster produces, not what the host produces.

Two claims are kept separate because they fail differently:

- **Structural**: device enumeration, topology, capabilities, collective
  numerics (the right bytes).
- **Timing**: latency and bandwidth of compute and of communication.

The project is strong on structural CUDA in one process, weak on compute timing
(`gpu-sim-u2o.17.4` parity FAIL: median relative error 0.97 / 0.41 / 0.47 for
elementwise / matmul / reduce), and absent on fabric.

## 2. What already exists (measured 2026-10-20)

| Piece | Where | State |
|---|---|---|
| CUDA runtime + NCCL interposer | `src/`, `shim/` | 156 exported symbols: 66 `cuda*`, 22 `cu*`, 37 `nccl*`; random fill, fixed/bandwidth latency, per-device pools (`MAP_PRIVATE|ANON`, single process) |
| Device-property overlay | `shim/cuda_runtime.h`, `tests/test_device_prop_overlay.cu` | real-layout values overlaid, `sizeof` guarded |
| Learned cost resolver | `include/fakegpu.h`, `GPUSIM_OP_LATENCY_JSON` | consumes a model artifact; no channel back |
| NCCL collectives | `src/fakegpu_nccl.c` | success + sleep; world size from env; **no topology awareness** |
| Net-transport plugin | `gpu-sim-465` (deferred) | ABI probe NO-GO: plugin dlopen'd, no member fired |
| Hardware | 1x RTX 5090 32GB, driver 615.78.08 | no second device |

The old NO-GO is explained, not unrecoverable. NCCL's net ABI is in the public
source: at tag `v2.30.7-1`, `src/include/plugin/nccl_net.h` and
`src/include/plugin/net/net_v{6..12}.h` define the structs, and
`src/plugin/net/net_v12.cc` shows the loader doing
`dlsym(lib, "ncclNetPlugin_v12")` on a **data** symbol. `gpu-sim-8g6` exported
symbols named `ncclNet_vN`; the loader's `dlsym` therefore found nothing, which
is exactly the recorded "LOADED, no member fired" signature. The route is
reopened as a source-vendoring task, not a binary reverse-engineering task.

## 3. Architecture (five layers)

```
 workload (torch / NCCL app / tests)
 ──────────────────────────────────────────────
 L4  real libnccl.so.2, unmodified              ← fidelity track
        │  ncclNetPlugin_v12 = our .so          ← inter-node seam
 ──────────────────────────────────────────────
 L3  fabric plugin: ncclNet over shm
 ──────────────────────────────────────────────
 L2  libfakecuda.so (driver API, cu*)           ← intra-node seam (P2P)
     libfakegpu.so (runtime + NCCL collectives) ← model track
 ──────────────────────────────────────────────
 L1  cluster substrate: topology spec, shm segment, rank registry, launcher
 ──────────────────────────────────────────────
 L0  one Linux box. No kernel module, no root, no second machine.
```

Two routes to fabric behaviour, deliberately both:

- **Model route (L2)** — topology + link model inside the collective
  interposer. Fast, deterministic, CI-usable, no real NCCL. Delivers
  "arbitrary networking setups" on its own.
- **Fidelity route (L3→L4)** — real NCCL algorithm executes; our net plugin
  and driver carry the bytes over simulated links. Higher fidelity, higher
  risk, and it is the **oracle** that tells us the model route is right.

Neither substitutes for the other: the model gives coverage and speed, the
plugin gives ground truth that the model's collective formulas are the ones
real NCCL actually implements.

Intra-node and inter-node are different seams in NCCL: intra-node data moves
through the CUDA driver (P2P copies, IPC handles) and inter-node data through
the net plugin. So per-link realism lives in `libfakecuda` for NVLink/PCIe and
in `libncclNet` for IB/RoCE/TCP. Both read the same topology spec.

## 4. The decision that makes timing honest

Absolute cluster timing inherits the compute-model error. Cluster acceptance
is therefore **difference-based**: hold the compute fixed, change only the
topology (or link parameters), and require the *delta* — the communication
component — to match prediction. Compute bias cancels. Absolute end-to-end
numbers may still be reported, but labelled as model-composed and carrying the
inherited error.

## 5. Route decisions and rationale

- **D1. Network route = both.** Model first for user-visible value; plugin
  first for de-risking, because the ABI is now a source-reading task. If the
  plugin fails again, the model route stands alone; if it succeeds, it is the
  oracle.
- **D2. Real NCCL is hosted, not reimplemented.** Load the real
  `libnccl.so.2` and place our code at its two seams (driver API below it,
  net plugin inside it). The collective-API fake remains for CI and is not the
  fidelity path.
- **D3. No kernel module, no root, no real multi-machine.** Device memory is
  host memory in shared segments; links are IPC.
- **D4. The topology spec is the single source of truth** for device identity,
  device model, links and parameters, and rank→device mapping. Nothing else
  defines topology.
- **D5. Shadow mode is P2.** Forwarding structural queries to the hosted real
  driver keeps the ABI honest, but arbitrary device counts cannot come from one
  real GPU. It is a validation aid, not a foundation.

## 6. Definition of done (epic gate `gpu-sim-cl1.10`)

1. Two ranks, real NCCL, our net plugin + fake driver: allreduce numerics equal
   a CPU reference (bytes, not fabricated).
2. A topology sweep (2/4/8 devices over NVLink, PCIe, 25GbE) produces the
   **predicted ordering** of allreduce time, and the delta between adjacent
   configs is within a pre-registered tolerance of the model.
3. Fault injection: a killed peer surfaces as a bounded NCCL error/timeout, not
   a hang.
4. `make check` green with no GPU present via the model route; structural
   counters (device count, world size, per-rank device id) correct for every
   config in the sweep.

## 7. Non-goals and named unknowns

- Kernel-level instruction timing fidelity — owned by `gpu-sim-u2o`, currently
  failing; inherited, not hidden.
- Real NIC/IB hardware emulation; RDMA is approximated by shm copies.
- Multi-machine orchestration.
- Unknown: whether the installed `libnccl` v12 layout matches tag `v2.30.7-1`
  exactly. Version detection at runtime is a bead acceptance criterion, not an
  assumption.
- Unknown: NCCL decides intra-node vs inter-node from its topology graph, not
  from plugin availability. Two fake devices in one process may both count as
  intra-node, so the net-plugin seam may never be reached. `gpu-sim-m3d.8`
  resolves this first (fake PCI domains + `NCCL_TOPO_FILE`, else the P2P seam)
  and the gate requires one config that provably uses NET.
