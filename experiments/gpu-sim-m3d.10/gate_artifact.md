# Gate artifact: gpu-sim-m3d.10 (Cluster acceptance gate)

## Verdict (pre-registered, fixed before this artifact)

**NO-CLOSE ON RESOLVABILITY.**

Two independent grounds, either sufficient:

### G1 — Resolvability (environmental)

4b requires the across-run spread of the two configs' medians, summed, to be at
most 1/10 of the delta being checked. For M1->M2 at 2 MiB with the corrected
model that is 36.2 us (d_pred 362.3 us). This box delivers tens of ms of
across-run spread (45985 us measured, committed variance artifact under
experiments/gpu-sim-pkb/). The requirement misses by ~3 orders of magnitude.

This is a finding about the MEASUREMENT ENVIRONMENT on a 2-CPU box at load
~20-40, NOT about the model and NOT about the fabric. The scheduler granularity
(1-4 ms) is 50-200x larger than the tightest pair's 4b allowance.

### G2 — Admissibility (unmeasured input)

The 4-device configs M3 and M4 are REFUSED for want of a MEASURED
pieces-per-phase at n=4. The model refuses rather than deriving P from the
fabric's cap, because the cap-based derivation undercounts the piece count by
four at 2 MiB. That measurement is gpu-sim-8ii, blocked by the 6kp interposer
increment (4 ranks on one GPU dies in ncclLaunchKernel; the interposer gives
device identity but not computation). The config set is NOT YET ADMITTED, and
the refusal appears here as a refusal with its reason, never as an estimate.

## Corrected per-pair predictions (P_actual chunk rule, master 273086b state)

Config set: mixed (M1, M2, M3, M4) at S = 2 MiB, admitted at 32 MiB and
re-admitted at 2 MiB.

| Pair | d_pred (us) | 4a bw_share | 4b rel margin | max tolerable spread (us) |
|------|-------------|-------------|---------------|---------------------------|
| M1->M2 | 362.3 | 0.823 | 0.872 | 36.2 |
| M2->M3 | ~268 | (see pkb) | (see pkb) | ~26.8 |
| M3->M4 | (see pkb) | (see pkb) | (see pkb) | (see pkb) |

Predicted order: M1 < M2 < M3 < M4 (at 2 MiB with corrected P).

The model's P is four times too small at 2 MiB (old P=2, corrected P=8). The
corrected latency term makes deltas LARGER (M2->M3 grows from 207.8 us to
~268 us), which does NOT rescue resolvability (still 50-200x below this box's
noise floor). The model fix is a CORRECTNESS requirement, not a resolvability
rescue.

## Noise floor (with evidence)

- Spread: 45985 us (max-min across 5 iterations at 2 MiB, excluding init).
- Box state: nproc=2 (cgroup CPU quota ~2 cores), loadavg ~20-40, ~3274 tasks.
- Scheduler granularity: 1-4 ms on this box.
- Raw per-iteration logs: experiments/gpu-sim-pkb/raw/iters_2m.log,
  experiments/gpu-sim-pkb/raw/iters_4k.log (committed, not /tmp).
- Full variance artifact: experiments/gpu-sim-pkb/variance_measurement.md.

The 46 ms spread stands as the evidence. No quieter box, no repeated runs to
find a tight cluster, no re-rolled median. The variance measurement is the
finding, not a problem to be solved.

## Phase-payload table (canonical form, 2 MiB payload)

| S (bytes) | floats | isends/rank | distinct bytes | phases (digests) | per-phase pieces | per-phase bytes | total bytes/rank | byte factor |
|-----------|--------|-------------|----------------|------------------|------------------|-----------------|------------------|-------------|
| 4096      | 1024   | 2           | 4096           | 2                | 1                | 4096 (S)        | 8192             | 2.0 (2S)    |
| 65536     | 16384  | 8           | 16384          | 2                | 4                | 65536 (S)       | 131072           | 2.0 (2S)    |
| 262144    | 65536  | 8           | 65536          | 2                | 4                | 262144 (S)      | 524288           | 2.0 (2S)    |
| 524288    | 131072 | 16          | 32768          | 2                | 8                | 262144 (S/2)    | 524288           | 1.0 (S)     |
| 1048576   | 262144 | 16          | 65536          | 2                | 8                | 524288 (S/2)    | 1048576          | 1.0 (S)     |
| 2097152   | 524288 | 16          | 131072         | 2                | 8                | 1048576 (S/2)   | 2097152          | 1.0 (S)     |

At 2 MiB: P=8 (8 pieces per phase, each 128 KiB = S/2), 2 phases, total per
rank = S (1 * S, byte factor 1.0 at this size for the per-direction figure;
2.0 total across both directions). The model's byte factor (2.0 at n=2) counts
both directions. The per-phase payload is S/2 at 2 MiB.

Full table: experiments/gpu-sim-pkb/canonical_byte_table.md and
experiments/gpu-sim-pkb/phase_payload_table.md.

## Artifact share (c_fixed bound)

- c_fixed (bound, not measurement): 1-2 us.
- Per-piece term at P=8, n=2: 8 x 2 = 16 us.
- Fraction of M1->M2 delta (362.3 us): 4-8%.
- Fraction of 100 us 4a floor: 16%.
- Inside the 20% artifact cap; at the edge of the 10% target.
- NOT the binding constraint (the variance is).

8nt stays POST-gate: the per-piece term does not by itself prevent a pass on a
box that could resolve the delta.

## Per-criterion statement

| # | Criterion | Status | Cause if unmet |
|---|-----------|--------|----------------|
| 1 | Per-link timing (plugin spends wire time) | MET | pkb: fabric honors per-link bw+latency, fg_fab_transfer spends n/B+L |
| 2 | Difference-based deltas (predicted vs measured) | UNMET | Environment: 46 ms spread vs 36.2 us requirement |
| 3 | Ordering preserved (measured = predicted order) | UNMET | Environment: cannot resolve deltas |
| 4 | Bounded failure on killed peer | MET | m3d.8: EPEERDEAD asserted (tests/test_fabric.c) |
| 5 | Structural counters | MET | trace records per-transfer link id/bytes/delay |
| 6 | No-GPU CI green | MET | make check passes |
| 7 | Wire bytes match model (per-rank) | UNMET | Unmeasured input: n=4 P refused (gpu-sim-8ii) |
| 8 | Timestamp clause | MET | model predictions timestamped pre-run |
| 9 | Compute statement (identical compute cancels) | MET | stated explicitly; see R3 |

Criteria 1, 4, 5, 6, 8, 9 are categorical outcomes measurable on this box.
Criteria 2, 3 are unmet due to the environment (noise floor).
Criterion 7 is unmet due to an unmeasured input (n=4 P).

## Explicit statement

The DoD is unreachable HERE while the rest of the direction's criteria are met.
The fault matrix (gpu-sim-m3d.9) and the N-device route (gpu-sim-6kp) assert
CATEGORICAL outcomes -- an error code (EPEERDEAD vs ETIMEDOUT), a rank count,
the absence of libcuda from the maps -- which are INSENSITIVE to the scheduler
noise that defeats a 20-36 us timing comparison. The environmental limit is to
ONE of the epic's criteria (the gate's timing comparison), not to the direction.

## Forbidden (recorded for auditability)

No quieter box was used. No repeated runs to find a tight cluster. No re-rolled
median. No widening of 4b. No payload chosen to enlarge the delta. The variance
measurement stands as the evidence. A gate that passes because its noise was
sampled until it looked good proves nothing.

## Pointers

- Variance artifact: experiments/gpu-sim-pkb/variance_measurement.md
- Raw logs: experiments/gpu-sim-pkb/raw/iters_4k.log, iters_2m.log
- Canonical byte table: experiments/gpu-sim-pkb/canonical_byte_table.md
- Phase-payload table: experiments/gpu-sim-pkb/phase_payload_table.md
- pkb metrics: experiments/gpu-sim-pkb/metrics.json
- Admissibility: experiments/gpu-sim-m3d.10/admissibility.py, metrics.json
- n=4 measurement: gpu-sim-8ii (blocked by 6kp)
- Model: epic_cluster/link_model.py (corrected chunk rule)
- C model: src/fakegpu_nccl.c (FG_NCCL_REFUSED sentinel)
