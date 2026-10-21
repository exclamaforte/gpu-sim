# gpu-sim-vs2 Pre-Registration (written before any measurement)

Timestamp (UTC): 2026-10-21T12:55:03Z  (file mtime; precedes first session at 2026-10-21T12:55:49Z)

## Settings (exactly two, fixed before measurement)
- S1 = current 2-CPU quota: cpu.max = 200000 100000 (2 CPUs). Obtained by running in the
  lab's normal worker scope.
- S2 = ONE larger quota, fixed at 800000 100000 (8 CPUs) BEFORE the first measurement,
  chosen because it is the largest sensible value on a 32-core box under neighbour load.
  Obtained via: systemd-run --user --scope --property=CPUQuota=800% -- <command>
  (verified to set cpu.max = 800000 100000).

## Workload (fixed, identical to tfq)
Sum of squares over N=100000 integers (Python int loop), run 4 times per session
(1 warmup + 3 measured). Durations in microseconds. No tuning between settings.

## Statistic (identical to gpu-sim-m3d.10/verdict.py)
Drop first value (init/warmup), drop values > 1e9 us (timer artifacts), spread = max - min.
Agreement check: check_statistic_agreement.py (PASS, committed).

## Measurements
Three independent sessions per setting, interleaved in time:
S1, S2, S1, S2, S1, S2  (so neighbour drift cannot masquerade as a quota effect).

## Decision rule (fixed in advance, may not be revised)
S2 ADMITS iff its spread is <= 36.2 us in at least 2 of its 3 sessions
AND its median spread is <= 36.2 us.
S1 not admitting is already established (gpu-sim-tfq) and is not re-litigated.
- If S2 admits: the gate may be re-run under its ORIGINAL protocol at S2, with the
  environment change stated in the artifact.
- If S2 does not admit: the finding is that the limit is NOT the quota; next suspect
  is neighbour load / scheduler behaviour on the shared cores.

## Requirement (from m3d.10)
36.2 us = 362.3 us (tightest modelled delta M1->M2) / 10.

## What must be reported either way
Both settings' numbers, interleaving order and timestamps, throttle deltas
(nr_throttled, throttled_usec per session from cpu.stat), verdict ADMITTED-AT-S2
or NOT-A-QUOTA-EFFECT. A worse result at S2 reported as prominently as a better one.
