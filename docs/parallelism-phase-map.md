# Parallelism phase map (gpu-sim-bw5)

CPU-only model (corrected compute term including attention seq^2, gpu-sim-ooj).
Counts hardware-validated (5o4.7); times modelled.

## Flip boundaries

- At S=8192 on 8 GPUs: winner tp=2 cp=4 (73.77 ms) — CP gets credit for attention sharding.
- At S=32768: winner still tp=2 cp=4, but step time 453.81 ms; CP benefit grows with S.
- At N=16: winner tp=1 pp=2 cp=8 (S=8192) or tp=1 cp=8 dp=2 (S=32768). CP dominates at long context.
- At N=32: winner tp=1 pp=2 cp=16 (S=8192) or tp=2 cp=16 (S=32768).
- 70B on 8 GPUs: infeasible (memory). On 16/32 GPUs: tp=8 pp=2 (TP-dominated, memory-bound).
- Tie count at max-overlap floor: 3 for 8B N=8; 1 elsewhere (no ties at N>=16).

## Indifferent cells (tie set size > 1)
- N=8 8B: 3 configs tie at max-overlap floor (22 ties at full overlap per 5o4.5).
