# gpu-sim-m3d.9: Fault injection design

## Per-fault-class specification (decided before coding)

### 1. Loss (loss_ppm)
- Injection point: fabric (fg_fab_transfer in src/fakegpu_fabric.c)
- Mechanism: per-transfer drop with probability ppm/1e6 (xorshift seeded by GPUSIM_SEED ^ rank ^ link_id)
- Expected CODE: FG_FAB_EINVAL (or ncclSystemError via plugin) on dropped transfer
- Not a delay: loss means failure, not slower
- Negative control: break the RNG seed (use 0), show loss rate becomes deterministic/uniform (not seeded), test RED
- Statistical test: loss bound computed from declared p and n (binomial), not observed rate
- Refusal: none (reachable on this box)

### 2. Jitter (jitter_us)
- Injection point: fabric (fg_fab_transfer) + model (latency term)
- Mechanism: duration ~ U(base, base+jitter) (uniform, documented)
- Expected CODE: FG_FAB_OK (jitter is a delay variation, not an error)
- Tolerance: pre-registered, may not be widened after seeing numbers
- Negative control: set jitter=0, show no variance in duration, test RED
- Refusal: none (reachable on this box)

### 3. Partition (shared_group declared partitioned)
- Injection point: fabric (fg_fab_transfer, shared_group check)
- Mechanism: links in a shared_group declared partitioned after k transfers (deterministic k) -> subsequent transfers fail
- Expected CODE: ETIMEDOUT (slow/partitioned), NOT EPEERDEAD (dead peer)
- Negative control: remove the partition declaration, show transfers succeed, test RED
- Refusal: "network partition" cannot be a real packet filter on this box (one shm fabric). The stand-in is: shared_group partition after k transfers. Named as stand-in.

### 4. Timeout (hard deadline per transfer)
- Injection point: fabric (fg_fab_transfer, deadline check)
- Mechanism: exceed deadline -> ETIMEDOUT -> NCCL error
- Expected CODE: ETIMEDOUT
- No unbounded sleep or spin (wpw contract: bounded FUTEX_WAIT)
- Negative control: set deadline to 0, show every transfer times out, test RED
- Refusal: none (reachable on this box)

### 5. Loss bound (statistical)
- Computed from DECLARED p and n (binomial), never from observed rate
- Negative control: use observed rate instead of declared p, show bound changes, test RED

### 6. Dead peer vs slow peer
- Dead peer: EPEERDEAD (killpeer, already enforced)
- Slow/partitioned peer: ETIMEDOUT
- Returning "an error" (not the specific code) FAILS
- Negative control: swap EPEERDEAD/ETIMEDOUT in assertion, show test RED

## Refusals (classes this box cannot reach)
- Real network partition (packet filter): not available (one shm fabric). Stand-in: shared_group partition after k transfers.
- Real packet loss injection: not available. Stand-in: probabilistic drop in fg_fab_transfer.

## Injection point summary
| Fault | Injection point | Expected CODE | Negative control |
|-------|----------------|---------------|------------------|
| loss | fabric (fg_fab_transfer) | FG_FAB_EINVAL/ncclSystemError | break RNG seed |
| jitter | fabric + model | FG_FAB_OK (delay variation) | jitter=0 -> no variance |
| partition | fabric (shared_group) | ETIMEDOUT | remove partition decl |
| timeout | fabric (deadline) | ETIMEDOUT | deadline=0 -> all timeout |
| dead peer | fabric (peer_alive) | EPEERDEAD | swap codes in assertion |

## CPU-only, <=2 cores. No ts job.
