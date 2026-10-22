# GPU-Simulator Research Lab: Direction-Level Report

## Epic: gpu-sim-m3d — Arbitrary-Cluster Simulation

### Summary
This report consolidates the artifacts and findings from the gpu-sim-m3d epic,
which aimed to simulate arbitrary N-GPU clusters on a single box.

### Key Beads and Outcomes

1. **gpu-sim-6kp** (N fake devices via CUDA runtime interposer): CLOSED
   - Interposer provides device count/identity; real driver provides memory/topology/NVML
   - Growth list enforced (30 symbols, nm A-B)
   - Net-plugin over interposed NCCL verified
   - 0c witness: differential probe shows runtime reaches real driver via dlopen

2. **gpu-sim-pkb** (Per-link bandwidth/latency): CLOSED
   - Phase-payload table measured (1/4/8 pieces per phase)
   - Chunk rule from measured table, not fabric cap
   - 12 CPU tests pass with negative controls
   - Variance measurement: 45985 us spread (environmental, not fixable)

3. **gpu-sim-m3d.9** (Fault injection): CLOSED
   - Loss, jitter, partition, timeout through fabric + plugin
   - ELOSS/ETIMEDOUT/EPEERDEAD trichotomy
   - 6 negative controls with RED output
   - Model route fault semantics (5zv) added

4. **gpu-sim-m3d.10** (Cluster acceptance gate): CLOSED
   - Verdict: NO-CLOSE: NO CONFIG SET IS ADMISSIBLE ON THIS HARDWARE
   - Pre-run admissibility rejection (4b unsatisfiable: 36.2 us required, ~6.6 ms offset)
   - violation_factor: 1270.3

5. **gpu-sim-5zv** (Model-route fault semantics): CLOSED
   - predict() reports loss_factor, jitter mean/sd, deadline-exceedance
   - C mirror applies loss factor
   - 8 tests + 4 negative controls
   - AC3: jitter draws within +/-5% mean, +/-15% sd (N=1000)
   - AC4: setting 1 MET (two-sided); settings 2-3 REFUSED (one device)

6. **gpu-sim-8ii** (Pieces-per-phase at n>=3): REFUSED
   - No route on this box gives 4 ranks real computation
   - Model keeps REFUSING n>=3

7. **gpu-sim-eut** (Per-link bandwidth/latency, n=2): CLOSED
   - Verdict: FACTOR-1.0-ABOVE-512KiB (NOT 'NO-STEP', withdrawn by name)
   - Usable evidence: 512KiB / 1MiB / 2MiB, ranks agreeing, exact-multiple gate
     passing -> factor 1.00 (2/2 runs each)
   - Below 512KiB: INVALID in the table. Mechanism now known (see item 3 below):
     NCCL selects LL at those sizes and the plugin's LL path does not complete;
     at 64KiB even SIMPLE does not complete. These rows are INVALID, not unmeasured.
   - Cite: experiments/gpu-sim-pkb/{results.md, metrics.json, reduction_output.txt, traces/}

8. **gpu-sim-ggh** (n=2 phase-fusion byte factor): CLOSED
   - The n=2 phase-fusion byte factor is 1.0, not 2.0
   - Special case in src/fakegpu_nccl.c:402-408 and epic_cluster/link_model.py deleted
   - n>=3 untouched and REFUSED, never cap-derived
   - P comes from the measured phase-payload table
   - Enforced by tests/test_model_agreement.py (C==Python within 1e-12,
     negative control on the old 2*S)

9. **gpu-sim-bl5** (Value correctness at 384KiB/512KiB): NEEDS-REVIEW (submitted)
   - 384KiB HANG (GPU kernel never completes, ncclAllReduce returns success)
   - 512KiB VALUE MISMATCH in 1 of 2 runs (5.0 where 3.0 expected, exact
     double-add of two chunks); 1 of 2 runs VALUE OK
   - Three documentation edits complete; lint passes; awaiting reviewer

10. **gpu-sim-7vw** (Sim fabric hang below 512KiB): DEFERRED
    - Root cause undetermined after long investigation
    - Both LL and SIMPLE protocols hang at 64KiB (2/2 runs each)
    - Defect is size-dependent, not protocol-specific
    - CPU tests fifo + reg PASS with negative controls

11. **gpu-sim-0xq** (Emulator LL path): CLOSED (REFUSE)
    - Plugin declares it does not support LL (notice at init)
    - Model validity domain: >=512KiB on SIMPLE path
    - Run with NCCL_PROTO=Simple

### Key Findings

1. **Real driver IS loaded and used** (memory/topology/NVML); device identity from shim

2. **Variance is environmental** (scheduler granularity 1-4 ms >> 36.2 us requirement)

3. **Protocol selection is part of every measurement** (the day's largest methodological
   finding): NCCL picks LL/LL128/Simple per size from a cost model over the link
   numbers the plugin advertises via getProperties (committed 512KiB log: LL 18.6/6.0,
   Simple 40.4/12.0; two-term crossing ~255KiB). S=393216 runs proto LL and HANGS;
   S=524288 runs proto SIMPLE and COMPLETES. The eut boundary is therefore most likely
   the LL/SIMPLE boundary -- a plugin capability gap, not a fabric size limit -- and
   the LL/Simple boundary MOVES with the advertised bandwidth/latency. Every row in the
   byte table now carries the protocol that produced it.

4. **Emulator validity domain**: below the LL boundary the emulated collective does not
   complete; above it, byte counts are deterministic (three 512KiB traces: 32 isends,
   32 recv-done, bytes=32768) while value checks have shown nondeterministic corruption
   in 1 of 2 runs (5.0 where 3.0 was expected, an exact double-add of two chunks).
   So: byte factors are trustworthy on the SIMPLE path; data correctness is not
   certified anywhere yet, and every value check so far used a uniform fill that
   cannot detect a misdelivery.

5. **Gate closes NO-CLOSE on admissibility**, not post-run failure

6. **Fault injection trichotomy enforced and demonstrated**

7. **Model route and plugin route agree on categorical outcomes**

### Not-Claimed Boundaries
- One real CUDA device (RTX 5090 index 0) so two-sided multi-rank fault measurement is
  refused at any rank count
- m3d.10 NO-CLOSE stands (no gate run)
- 4tw validates the model against OUR fabric, not hardware loss
- n>=3 pieces-per-phase REFUSED

### Artifacts
- experiments/gpu-sim-6kp/: probe outputs, maps, growth list, run.sh
- experiments/gpu-sim-pkb/: phase payload table, canonical byte table, variance
  measurement, tests, traces/
- experiments/gpu-sim-m3d.9/: design, fault tests, negative controls
- experiments/gpu-sim-m3d.10/: gate artifact, verdict.py, admissibility audit
- experiments/gpu-sim-5zv/: model fault tests, negative controls
- experiments/gpu-sim-eut/: regenerated byte table (77d), PROTOCOL column
