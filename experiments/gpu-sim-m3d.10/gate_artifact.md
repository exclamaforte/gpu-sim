# Gate artifact: gpu-sim-m3d.10 (Cluster acceptance gate)

## Verdict (pre-registered, fixed before this artifact)

**NO-CLOSE: NO CONFIG SET IS ADMISSIBLE ON THIS HARDWARE.**

The verdict WORD is NO-CLOSE (pre-registered at 07:51 before the artifact existed).
The GROUND is refined from "resolvability" to "admissibility" because 4b is a
pre-run constraint: its 10x-spread clause cannot be satisfied by any spread this
box delivers (36.2 us required, tens of ms available), so no config set is
admissible here and the gate does not run. OUTCOME UNCHANGED. EVIDENCE UNCHANGED.
Refinement stated because it is more conservative: a pre-run rejection is
stronger than a post-run failure.

This is an ADMISSIBILITY REJECTION, not a measured outcome. 4b is a pre-run
constraint ("or the config set is REJECTED before the run"). Its 10x-spread
clause cannot hold here (36.2 us needed, tens of ms available), so NO CONFIG
SET IS ADMISSIBLE HERE. The gate does NOT RUN. Criteria 2, 3, and 9(a)-(c)
are NOT APPLICABLE (not UNMET).

Two independent grounds, either sufficient:

### G1 — Admissibility (pre-run: no config set is admissible)

The reading is present but not resolvable: ~6.6 ms offset (minima 6677 us @4 KiB
vs 6717 us @2 MiB) against model predictions of 53.3 us (M1) and 415.5 us (M2),
so the payload response is unresolvable through a term ~150x larger than the
predicted response. 4b's 36.2 us requirement is ~200x below the reading's
granularity. The measured per-iteration time does not respond to the payload
within the ~6.6 ms per-iteration term (environment/harness, not model). The timed
window is the collective window, not process wall-clock (init iteration 164257 us
excluded).

This ~6.6 ms payload-independent term is the EVIDENCE FOR THE REJECTION, not a
failed gate measurement.

### Timeline note (refinement recorded)

Pre-registered at 07:51 as "NO-CLOSE on resolvability" (outcome fixed before the
artifact existed). Refined to the admissibility form on discovering that 4b is a
pre-run constraint, not a post-run test. OUTCOME UNCHANGED (NO-CLOSE). EVIDENCE
UNCHANGED (same spread numbers, same box state, same raw logs). Refinement stated
because it is more conservative: a pre-run rejection is stronger than a post-run
failure, and a reader must see both what we changed and why.

### G2 — Admissibility (unmeasured input)

The 4-device configs M3 and M4 are REFUSED for want of a MEASURED
pieces-per-phase at n=4. The model refuses rather than deriving P from the
fabric's cap, because the cap-based derivation undercounts the piece count by
four at 2 MiB. That measurement is gpu-sim-8ii, blocked by the 6kp interposer
increment. The config set is NOT YET ADMITTED, and the refusal appears here as
a refusal with its reason, never as an estimate.

## Corrected per-pair predictions (P_actual chunk rule, master state)

Config set: mixed (M1, M2, M3, M4) at S = 2 MiB, admitted at 32 MiB and
re-admitted at 2 MiB.

| Pair | d_pred (us) | 4a bw_share | 4b rel margin | max tolerable spread (us) |
|------|-------------|-------------|---------------|---------------------------|
| M1->M2 | 362.3 | 0.823 | 0.872 | 36.2 |
| M2->M3 | ~268 | (see pkb) | (see pkb) | ~26.8 |
| M3->M4 | (see pkb) | (see pkb) | (see pkb) | (see pkb) |

Predicted order: M1 < M2 < M3 < M4 (at 2 MiB with corrected P).

The model's P is four times too small at 2 MiB (old P=2, corrected P=8). The
corrected latency term makes deltas LARGER, which does NOT rescue resolvability
(still 50-200x below this box's noise floor). The model fix is a CORRECTNESS
requirement, not a resolvability rescue.

## Noise floor (proxy, not protocol measurement)

- Proxy spread: 45985 us (max-min across 5 iterations at 2 MiB, excluding init).
  Sample: n=5, one config (2 MiB), one run reached 52702 us. This is a PROXY
  for 9(a)'s protocol (max-min over exactly 3 runs/config), which was NOT
  executed. The 4641 s timer artifact (raw/iters_4k.log) is EXCLUDED from this.
- Box state: nproc=2 (cgroup CPU quota ~2 cores), loadavg ~20-40, ~3274 tasks.
- Scheduler granularity: 1-4 ms on this box.
- Raw per-iteration logs: experiments/gpu-sim-pkb/raw/iters_2m.log,
  experiments/gpu-sim-pkb/raw/iters_4k.log (committed, not /tmp).
- Full variance artifact: experiments/gpu-sim-pkb/variance_measurement.md.

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
rank = S (byte factor 1.0 per direction; 2.0 total across both directions).

## Per-criterion statement (bead's own acceptance criteria)

| # | Criterion | Status | Cause if not met |
|---|-----------|--------|------------------|
| 1 | metrics.json + committed verdict script | MET | committed, runs, fails loudly on missing input |
| 2 | Ordering (measured = predicted order) | NOT-APPLICABLE | no admitted config set (4b rejected) |
| 3 | Delta error vs 35% | NOT-APPLICABLE | no admitted config set |
| 4a | bw_share (\|d_bw\| >= 0.5*\|d_pred\|) | MET (at 2 MiB: 0.823) | pre-run admissibility, computed from model |
| 4b | 100 us floor | MET (362.3 us > 100) | pre-run admissibility |
| 4b | 20% relative margin | MET (0.872) | pre-run admissibility |
| 4b | 10x spread | UNMET (environment) | 36.2 us needed, tens of ms available |
| 4c | Cancellation statement | MET | compute identical across configs, cancels in difference |
| 4d | Config set re-established vs corrected model | MET | admitted 'mixed' set at 2/8/32 MiB |
| 5 | Structural counters | MET (on M1/M2) | NOT-APPLICABLE beyond M1/M2 (n=4 refused) |
| 6 | Kill test (EPEERDEAD) | MET | m3d.8 evidence, bounded survivor exit |
| 7 | No-GPU lane (clean clone) | MET | make check passes |
| 8 | Explicit CLOSE/NO-CLOSE | MET | this artifact |
| 9a | 3 runs/config, median, spread | NOT-APPLICABLE | no GPU run (CPU-only, no admitted set) |
| 9b | delta_measured = median_i - median_j | NOT-APPLICABLE | no GPU run |
| 9c | ordering on medians, unresolved pairs | NOT-APPLICABLE | no GPU run |
| 9d | measured wire bytes re-derived from trace | MET (M1/M2), NOT-APPLICABLE (M3/M4) | n=4 refused (gpu-sim-8ii) |
| 9e | real-GPU runs through ts with job id | NOT-APPLICABLE | CPU-only, no GPU run |
| 9f | verdict by committed script, loud failure on missing input | MET | verdict.py exists, refuses exit 2 on missing input |

### Arbiter's criteria (separate section)

The arbiter's criteria 1-9 (per the review) are tracked separately. The bead's
own acceptance criteria above are the authoritative list for this gate.

## c_fixed (bound, not measurement)

- c_fixed (bound): 1-2 us.
- Per-piece term at P=8, n=2: 16-32 us (16 pieces/rank).
- Cancels EXACTLY in M1->M2 (both n=2, 16 pieces/rank, chunk count speed-independent).
  Artifact share of the admitted delta: 0 us / 0%.
- Absolute 10% band: 16-32 us is 30-60% of M1's own 53.3 us.
- NOT the binding constraint (the variance is).

## Explicit statement

The DoD is unreachable HERE while the rest of the direction's criteria are met.
The fault matrix (gpu-sim-m3d.9) and the N-device route (gpu-sim-6kp) assert
CATEGORICAL outcomes -- an error code (EPEERDEAD vs ETIMEDOUT), a rank count,
the absence of libcuda from the maps -- which are INSENSITIVE to the scheduler
noise that defeats a 20-36 us timing comparison. The environmental limit is to
ONE of the epic's criteria (the gate's timing comparison), not to the direction.

## Forbidden (recorded for auditability)

No quieter box was used. No repeated runs to find a tight cluster. No re-rolled
median. No widening of 4b. No switching to a payload chosen because it made the
delta bigger. The variance measurement stands as the evidence.

## What was NOT done (a negative artifact states its own negatives)

- NO GATE RUN was performed.
- NO MEASURED DELTAS were produced.
- NO 3-RUN-PER-CONFIG PROTOCOL was executed.
- NO GPU RUN was performed (CPU-only).
- NO ORDERING was OBSERVED (measured or otherwise).

This artifact reports a pre-run admissibility rejection. It does not contain
measured data, observed orderings, or run results. Any language in this artifact
implying otherwise ("measured delta", "measured ordering", "we ran", "observed")
would be incorrect.

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
