# Direction Report: gpu-sim-m3d — Arbitrary-Cluster Simulation on One Box

## 1. The Question

Simulate an arbitrary N-GPU cluster with settable fabrics on one box, and decide
by measurement whether a proposed timing criterion (m3d.10) is achievable.

## 2. What Was Built (dependency order)

| # | Bead | What | Proving artifact |
|---|------|------|-----------------|
| 1 | 6kp | Real libnccl over a modelled shm fabric via CUDA runtime interposer | experiments/gpu-sim-6kp/probe_nccl4_output.txt (4-rank init only; kernel launch does not follow -- see Limit 2) |
| 2 | wpw | Fabric spends modelled wire time (bw + latency); futex wakeup; EPEERDEAD/ETIMEDOUT | src/fakegpu_fabric.c (bounded FUTEX_WAIT; wake on slot publish/release) |
| 3 | pkb | Per-link bw/latency in plugin data path; measured phase-payload table | experiments/gpu-sim-pkb/phase_payload_table.md |
| 4 | m3d.7 | Topology-aware collectives in NCCL interposer + JSONL trace | src/fakegpu_nccl.c (fg_nccl_predict) |
| 5 | m3d.8 | Net plugin over shm fabric (2-rank real-NCCL numerics) | experiments/gpu-sim-m3d.8/scenario_normal.log |
| 6 | m3d.9 | Fault injection: loss/jitter/partition/timeout (ELOSS/ETIMEDOUT/EPEERDEAD) | experiments/gpu-sim-m3d.9/design.md + test_fault.c |
| 7 | 5zv | Model-route fault semantics (1/(1-p), deadline-exceedance, predicted mean/sd) | epic_cluster/link_model.py + tests/test_model_fault.py |
| 8 | m3d.10 | Cluster acceptance gate | experiments/gpu-sim-m3d.10/gate_artifact.md |

### Key design decisions
- **Pieces-per-phase from MEASURED table, not fabric cap**: the cap-derived estimate
  undercounts the piece count by four at 2 MiB (gate_artifact.md, G2;
  admissibility_pi_audit.json). No cap-based fallback anywhere.
- **n>=3 refused, not estimated**: the model raises ValueError rather than deriving P from the cap.
- **Chunk rule is an INPUT** (NCCL protocol property), not a modelled function.

## 3. The Verdict

**NO-CLOSE: NO CONFIG SET IS ADMISSIBLE ON THIS HARDWARE.**
(experiments/gpu-sim-m3d.10/gate_artifact.md, verdict.py exit 1, violation_factor 1270.3)

Two independent grounds:
1. **Admissibility (G1)**: 4b's 10x-spread clause requires 36.2 us; this box delivers
   ~6.6 ms per-iteration term (minima 6677 us @4 KiB vs 6717 us @2 MiB).
   The payload response (53.3 us M1, 415.5 us M2) is unresolvable through a term
   ~150x larger.
2. **M3/M4 refused (G2)**: n=4 pieces-per-phase unmeasured (8ii refused).

Asymmetric-error argument: a pre-run rejection is stronger than a post-run failure.
The gate does NOT run. Criteria 2, 3, 9(a)-(c) are NOT APPLICABLE.

## 4. The Three Limits

1. **Timing resolvability** (environmental): the 36.2 us requirement is not
   achievable on this box, and the mechanism is now named, not ambient.
   (a) tfq: no admissible operating point across 16 windows at a 2-CPU quota (min spread
   53.4 us vs the 36.2 us requirement; experiments/gpu-sim-tfq/windows.jsonl +
   spread_vs_load.md).
   (b) vs2: the quota IS a real determinant -- raising it from 2 to 8 CPUs cut the
   median spread 7.7x (1072.2 -> 139.9 us) and the best window 16x (782.4 -> 47.9 us)
   with zero throttle deltas, so the mechanism is cgroup parallelism, not throttling
   (experiments/gpu-sim-vs2/metrics.json + session_*.json).
   (c) ep3: at core granularity no quiet core -- least-busy of 32 cores was 0.3584
   (0/32 below 0.25), NO-QUIET-CORE by pre-registration
   (experiments/gpu-sim-ep3/selection.json + metrics.json).
   (d) 3ba: a quiet core DOES exist (cpu 8 at 0.2194) and 2 of 3 windows met the
   requirement (15.2, 25.3 us) -- but the instrument self-charged (200 ms span
   containing the window's own ~25 ms CPU), so the pre-registered verdict was
   NO-ADMISSION-QUIET-CORE (experiments/gpu-sim-3ba/results.md).
   (e) 99g: corrected instrument (util excluding own CPU) worked -- other_util
   0.18/0.17/0.10 -- yielding 1 of 3 qualifying windows (35.1 us), but the SMT
   sibling (cpu 24) was saturated (0.94-1.0), explaining the bimodality
   (experiments/gpu-sim-99g/results.md + metrics.json).
   (f) 52i: no both-siblings-quiet physical core in 20 samples (best-pair worst
   0.2500, 0/20 below 0.25; experiments/gpu-sim-52i/samples.jsonl).
   (g) y1h: 200 ms average cannot resolve ~6 ms interference events (12x spread
   variation within one 200 ms band; rho=-0.0035, RANGE-INSUFFICIENT;
   experiments/gpu-sim-y1h/results.md).
   (h) r27: SMT sibling contention at run scale -- Spearman rho=0.5158 across 36
   runs; mean duration 3570.1 us (sibling quiet) vs 5066.6 us (sibling busy),
   +42% penalty; occupier is EXTERNAL (ca/codds-ca 99.2%, python3/constlaw 97.6%,
   yes/cclm 71.8%); 0 of 36 snapshots were our own lab
   (experiments/gpu-sim-r27/metrics.json + windows.jsonl).
   Conclusion: the criterion is not unachievable in principle; it is unobtainable
   on this box because it needs an exclusively-owned physical core, and 16 cores
   shared with five research labs do not supply one. What would answer it: an
   exclusive core / isolated cpuset, a dedicated machine, or a hardware-isolated
   measurement path.
   Artifact: experiments/gpu-sim-pkb/variance_measurement.md (45985 us spread).

   **Instrument lineage** (each bead measured a different resource; the sequence is
   the methodological finding):
   - ep3: measured the BOX (whole-system load) -- wrong resource.
   - s9j: measured the USER SLICE (all labs' activity) -- wrong resource.
   - 3ba/99g: measured a LOGICAL CPU -- wrong resource (SMT sibling).
   - 52i: measured the PHYSICAL CORE (both siblings) -- correct resource, but none
     quiet in 20 samples.
   - y1h: measured a 200 ms AVERAGE -- too coarse (12x variation within one band).
   - r27: measured the INDIVIDUAL RUN -- correct scale, and found the mechanism.

   **Not claimed**: no admission anywhere in this thread; 3ba's and 99g's
   sub-requirement windows are not evidence of an achievable operating point;
   m3d.10's NO-CLOSE-ADMISSIBILITY stands unchanged (no gate run occurred).
   The timing limit is environmental, not a simulator defect.
2. **n>=3 pieces-per-phase**: the interposer gives device IDENTITY, not per-rank
   COMPUTATION (nvidia-smi shows 1 device). 8ii closed as REFUSED.
   Artifact: experiments/gpu-sim-6kp/real_driver_witness.txt.
3. **Per-logical-transfer fidelity**: deferred (8nt) because it would invalidate the
   per-isend accounting every measured artifact rests on.

## 5. What to Conclude (and Not)

- **Test-enforced properties**: fault trichotomy (ELOSS/ETIMEDOUT/EPEERDEAD), chunk
  rule refusal, growth list enforcement, net-plugin assertion, guard uniqueness.
  All have demonstrated RED runs committed.
- **Environmental findings**: timing variance, gate verdict. Not fixable by code.
- **The model refuses n>=3 rather than estimating it.** No cap-derived fallback exists.
- **Categorical results are insensitive to scheduler noise** (m3d.9 scope note).
- **Pre-registration discipline** produced three corrections in this direction: the
  98%-throttle misattribution to the wrong lab's scope, the vs2 mechanism refutation,
  and the ep3 n=1 limitation. That discipline is why the numbers above can be trusted.

## 6. Reproduction (verified from clean checkout)

```bash
# Growth list enforcement
make check
# Expected: "PASS: growth list matches (A-B) exactly (30 entries)"

# Net-plugin assertion
python3 experiments/gpu-sim-6kp/test_net_plugin.py
# Expected: "PASS net_plugin_loaded"

# Model fault tests
python3 tests/test_model_fault.py
# Expected: "ALL MODEL FAULT TESTS PASS"

# Fabric tests
make test-fabric
# Expected: "fabric tests PASS"

# Gate verdict (PI-written judge; exit 1 = failing verdict, a reported result)
python3 experiments/gpu-sim-m3d.10/verdict.py experiments/gpu-sim-m3d.10/metrics.json
# Expected: VERDICT: NO-CLOSE-ADMISSIBILITY ...; JSON has violation_factor=1270.3; exit code 1
```

## Verification list (spot-checked citations)

1. 30 growth-list entries: nm -D --undefined-only libnccl minus nm -D --defined-only libfakegpu (scripts/gen_growth_list.py --check)
2. 4-rank init PASS: experiments/gpu-sim-6kp/probe_nccl4_output.txt
3. violation_factor 1270.3: experiments/gpu-sim-m3d.10/verdict_out.json
4. 6 negative controls RED: experiments/gpu-sim-m3d.9/negctrl/
5. 51 of 442 covered: experiments/gpu-sim-6kp/real_driver_witness.txt (nm counts)
6. 2.3% mean error, 0.4% sd error: experiments/gpu-sim-5zv/jitter_draws_comparison.md (N=1000)
7. SMT sibling contention: rho=0.5158 (36 runs), +42% duration penalty, external occupiers
   (ca/codds-ca 99.2%, python3/constlaw 97.6%, yes/cclm 71.8%):
   experiments/gpu-sim-r27/metrics.json + windows.jsonl

## 7. Corrected Conclusions (gpu-sim-x70)

### 7.1 eut (closed): FACTOR-1.0-ABOVE-512KiB, NOT 'NO-STEP'

Verdict: FACTOR-1.0-ABOVE-512KiB (withdrawn 'NO-STEP' by name). Usable evidence:
512KiB / 1MiB / 2MiB, ranks agreeing, exact-multiple gate passing -> factor 1.00
(2/2 runs each). Below 512KiB: INVALID in the table; the MECHANISM is now known
(see 7.3) rather than merely unknown. Cite: experiments/gpu-sim-pkb/{results.md,
metrics.json, reduction_output.txt, traces/}.

### 7.2 ggh (closed): n=2 phase-fusion byte factor is 1.0, not 2.0

The special case in src/fakegpu_nccl.c:402-408 and epic_cluster/link_model.py is
deleted; n>=3 untouched and REFUSED, never cap-derived; P comes from the measured
phase-payload table; enforced by tests/test_model_agreement.py (C==Python within
1e-12, negative control on the old 2*S).

### 7.3 Protocol selection is part of every measurement (the day's largest
methodological finding)

NCCL picks LL/LL128/Simple per size from a cost model over the link numbers the
plugin advertises via getProperties (committed 512KiB log: LL 18.6/6.0, Simple
40.4/12.0; two-term crossing ~255KiB). S=393216 runs proto LL and HANGS;
S=524288 runs proto SIMPLE and COMPLETES. The eut boundary is therefore most
likely the LL/SIMPLE boundary -- a plugin capability gap, not a fabric size limit
-- and the LL/Simple boundary MOVES with the advertised bandwidth/latency. State
that every row must carry the protocol that produced it. Every row in the byte
table now carries a PROTOCOL column (gpu-sim-77d, commit 713f4e3).

### 7.4 Emulator validity domain

Below the LL boundary the emulated collective does not complete; above it, byte
counts are deterministic (three 512KiB traces: 32 isends, 32 recv-done,
bytes=32768) while value checks have shown nondeterministic corruption in 1 of 2
runs (5.0 where 3.0 was expected, an exact double-add of two chunks). So: byte
factors are trustworthy on the SIMPLE path; data correctness is not certified
anywhere yet, and every value check so far used a uniform fill that cannot detect
a misdelivery. The two decodes that prove the corruption is an offset error
rather than noise:
  101.304006 = 3.0 + 0.001*98304 (a contribution from offset 0.75S landing at
  index 0)
  134.072006 = 1.0 + (2.0 + 0.001*131072) (an element one past the end)
These only became visible with a non-uniform fill.

### 7.5 Not-claimed boundaries

One real CUDA device (RTX 5090 index 0) so two-sided multi-rank fault measurement
is refused at any rank count; m3d.10 NO-CLOSE stands (no gate run); 4tw
validates the model against OUR fabric, not hardware loss; n>=3 pieces-per-phase
REFUSED.

### 7.6 The reference-failure error class

Six-plus instances in one day, each caught by someone other than its author -- the
byte table's /tmp traces, digest-as-phase-marker, the lint counting .md as an
artifact, the model's 2S constant, the PI's bl5 bead description, sum[0]=3.0
read as a 384KiB result when the harness hardcodes n=1024, and a truncated NCCL
log read as 'NCCL stuck at Connect'. Name the mechanical controls now in place:
scripts/check_prose_numbers.py, scripts/check_bead_prose_numbers.py, per-row
proximity mode, ratio/integer grammar, tests/test_model_agreement.py,
tests/test_fabric.c. State the durable rule: before citing a result, open the
code or log that produced it and confirm the SIZE, the SUBJECT and the HARNESS.
