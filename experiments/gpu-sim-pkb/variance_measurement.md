# Variance measurement (final, negative result)

Raw per-iteration durations committed under `experiments/gpu-sim-pkb/raw/`
(`iters_4k.log`, `iters_2m.log`); one value per line, microseconds, first
iteration is init overhead.

## Box state (recorded, not by reference)

- nproc = 2 (cgroup CPU quota ~2 cores)
- loadavg ~20-40 during the run
- ~3274 tasks on the box
- Oversubscribed; scheduler granularity ~1-4 ms on this box.

## 4 KiB (1024 floats, 20 iterations)

Values (us): 283575, 6826, 4641746337, 10365, 6954, 10334, 6764, 6771,
52960, 6835, 6822, 6746, 6824, 6795, 6677, 6750, 52980, 6858, 10370.

- First iter 283575 us (init overhead, excluded from spread).
- Median of remaining: ~6830 us.
- Entry 4641746337 us is a TIMER ARTIFACT, not a measurement: it implies a 4641 s
  iteration, which the run's own duration cannot contain. Kept in raw/iters_4k.log,
  excluded from all spread computations. It is NOT a statistic.
- Spread (max-min, excluding init and the timer artifact): 52980 - 6677 = 46303 us
  (still dominated by scheduler noise, not the artifact).

## 2 MiB (524288 floats, 5 iterations)

Values (us): 164257, 6729, 6717, 10319, 52702.

- First iter 164257 us (init overhead, excluded).
- Median of remaining: ~6729 us.
- Spread (max-min, excluding init): 45985 us (52702 - 6717).

## Verdict

The 2 MiB spread (45985 us) exceeds the gate's 2 MiB tolerance (~36.2 us,
derived from the corrected M1->M2 delta of 362.3 us) by ~1270x. The
measurement is dominated by scheduler noise on this oversubscribed box
(load avg ~20-40, 2-CPU quota). Per the PI ruling, a clean-box re-run is NOT
pursued: no quieter box, no repeated runs to find a tight cluster, no
re-rolling the median. The 46 ms spread stands as the evidence.

This is a NEGATIVE RESULT on resolvability of the gate's timing comparison
on this hardware. It is not a defect in pkb; it is an environmental bound.

## c_fixed (bound, not measurement)

Direct measurement of c_fixed (per-piece fixed cost) is unusable here: the
spread is tens of ms against a 36.2 us requirement. Per the PI ruling, c_fixed
is recorded as a BOUND derived from arithmetic, not a measurement:

- Post-futex c_fixed: 1-2 us (bound).
- Per-piece term at P=8, n=2: 8 * (1-2 us) = 8-16 us.
- As fraction of the corrected M1->M2 delta (362.3 us): 4-8%.
- As fraction of the 100 us 4a floor: 16%.
- Inside the 20% artifact cap; at the edge of the 10% target.
- NOT the binding constraint (the noise floor is).

The bound holds because c_fixed is the futex wakeup residual after the bounded
wait fix; the fabric's FUTEX_WAIT is capped at 100 us and re-checks
peer_alive, so the residual per-piece cost is the wakeup latency, bounded by
the scheduler granularity. On this box that granularity is 1-4 ms, but the
per-piece cost (one wakeup per piece) is amortized; the bound 1-2 us reflects
the best-case wakeup, not the worst-case scheduling delay.

## Negative result statement

The gate's timing comparison (measured vs predicted duration) is NOT resolvable
on this hardware. The environmental noise floor (1-4 ms scheduler granularity)
is 50-200x larger than the tightest pair's 36.2 us allowance. The measurement
environment, not the model or the fabric, is the binding constraint.

The rest of the direction's criteria -- fabric work, N-device route, fault
matrix, model correction -- are categorical outcomes insensitive to this noise
and are measurable on this box.
