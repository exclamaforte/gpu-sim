# gpu-sim-4tw: Model vs fabric under loss

## Verdict: MODEL-SUPPORTED

All 5 test points (p = 0, 0.05, 0.10, 0.25, 0.50) have median attempts/deliveries within 5% of 1/(1-p).

| loss_ppm | p | median ratio | model 1/(1-p) | rel err | within? |
|---|---|---|---|---|---|
| 0 | 0.00 | 1.000 | 1.000 | +0.0000 | yes |
| 50000 | 0.05 | 1.047 | 1.053 | -0.0054 | yes |
| 100000 | 0.10 | 1.092 | 1.111 | -0.0172 | yes |
| 250000 | 0.25 | 1.314 | 1.333 | -0.0145 | yes |
| 500000 | 0.50 | 2.005 | 2.000 | +0.0025 | yes |

## Instrument check
PASSED: loss=0 -> 100/100; loss=1e6 -> 100/0 (attempts/successes).

## Interpretation
The fabric's per-transfer loss (returning FG_FAB_ELOSS and retrying) yields a geometric
retransmit distribution whose mean matches the model's closed form 1/(1-p). The model
route's loss counterfactual is validated against the simulator's own fault behaviour.
No mismatch mechanism was triggered; no model correction needed.
