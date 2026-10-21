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

### Key Findings
- Real driver IS loaded and used (memory/topology/NVML); device identity from shim
- Variance is environmental (scheduler granularity 1-4 ms >> 36.2 us requirement)
- Gate closes NO-CLOSE on admissibility, not post-run failure
- Fault injection trichotomy enforced and demonstrated
- Model route and plugin route agree on categorical outcomes

### Artifacts
- experiments/gpu-sim-6kp/: probe outputs, maps, growth list, run.sh
- experiments/gpu-sim-pkb/: phase payload table, canonical byte table, variance measurement, tests
- experiments/gpu-sim-m3d.9/: design, fault tests, negative controls
- experiments/gpu-sim-m3d.10/: gate artifact, verdict.py, admissibility audit
- experiments/gpu-sim-5zv/: model fault tests, negative controls
