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

1. **Timing resolvability** (environmental): scheduler granularity 1-4 ms is 50-200x
   larger than the 36.2 us requirement. The cause is not an unavailable quiet box: the
   per-run scope carries a 2-CPU cgroup quota (cpu.max 200000/100000) and is throttled
   (nr_throttled 64 of 2826 periods in one sampled scope; live scopes showed ~98% of
   periods throttled). Whether a larger quota admits is UNTESTED here.
   Artifact: experiments/gpu-sim-pkb/variance_measurement.md (45985 us spread);
   experiments/gpu-sim-tfq/windows.jsonl + cgroup finding.
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
