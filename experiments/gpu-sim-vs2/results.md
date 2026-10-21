# gpu-sim-vs2: Quota experiment

## Pre-registration

See pre_registration.md (timestamp 2026-10-21T12:55:03Z, before first measurement 12:55:49Z).

## Settings

- S1: cpu.max = 200000 100000 (2 CPUs)
- S2: cpu.max = 800000 100000 (8 CPUs), via systemd-run --user --scope --property=CPUQuota=800%

Interleaving: S1_1, S2_1, S1_2, S2_2, S1_3, S2_3

## Results

| session | quota | spread (us) | throttle delta | top non-cgroup consumers |
|---|---|---|---|---|
| S1_1 | 200000 100000 | 1072.2 | 0 | python3, python3, python3 |
| S1_2 | 200000 100000 | 1326.6 | 0 | python3, bd, ca |
| S1_3 | 200000 100000 | 782.4 | 0 | python3, ca, python3 |
| S2_1 | 800000 100000 | 293.2 | 0 | bd, python3, ps |
| S2_2 | 800000 100000 | 47.9 | 0 | python3, ca, python3 |
| S2_3 | 800000 100000 | 139.9 | 0 | bd, bd, bd |

## Summary

- S1 spreads: [782.4, 1072.2, 1326.6] us, median 1072.2 us
- S2 spreads: [47.9, 139.9, 293.2] us, median 139.9 us
- S2 sessions admitting (<=36.2): 0/3

## Verdict

**NOT-A-QUOTA-EFFECT** — NOT-A-QUOTA-EFFECT: S2 spreads [47.9, 139.9, 293.2] us, median 139.9 us; 0/3 sessions <= 36.2 us.

PI pre-registered prediction: NOT-A-QUOTA-EFFECT. Result: NOT-A-QUOTA-EFFECT (prediction CORRECT).

## Note on neighbour contention

Top non-cgroup consumers captured were lab tooling (python3, bd, ca, ps), not the heavy neighbour processes reported by the PI. The snapshot reflects average CPU since process start, not instantaneous. Neighbour contention remains plausible but is not conclusively evidenced by this snapshot.
