# gpu-sim-tfq: Spread vs Load (environment characterization)

## Pre-registered admission rule (verbatim)

4b requires the proxy spread to be at most 1/10 of the tightest modelled delta. An operating point ADMITS iff (i) measured spread <= absolute requirement, AND (ii) the measurement is repeatable: the same window, re-measured, stays inside the requirement, AND (iii) load is recorded and stays within the band claimed for that point. A point satisfying (i) but not (ii) is NON-ADMISSIBLE-FLUKY.

**Absolute requirement:** 36.2 us (tightest modelled delta 362.3 us / 10). Arithmetic: 362.3 us / 10 = 36.23 us (committed as 36.2 us)

## Workload (fixed)

Fixed CPU workload: sum of squares over N=100000 integers (Python int loop), run 4 times per window (1 warmup + 3 measured); durations in microseconds. No tuning between windows.

## Statistic reused

From experiments/gpu-sim-m3d.10/verdict.py: drop first value (init), drop >1e9 us timer artifacts, max-min. Agreement proven by check_statistic_agreement.py.

## Spread-vs-load table

| window | phase | loadavg1 | mean aff util | ctx sw | spread (us) | status |
|---|---|---|---|---|---|---|
| w01 | 1 | 19.76 | 0.517 | 2 | 53.4 | NOT-ADMITTED |
| w02 | 1 | 19.57 | 0.536 | 8 | 1986.6 | NOT-ADMITTED |
| w03 | 1 | 20.41 | 0.628 | 1 | 1733.8 | NOT-ADMITTED |
| w04 | 1 | 20.24 | 0.584 | 2 | 123.6 | NOT-ADMITTED |
| w05 | 1 | 23.13 | 0.745 | 2 | 1146.8 | NOT-ADMITTED |
| w06 | 1 | 23.09 | 0.548 | 7 | 355.1 | NOT-ADMITTED |
| w07 | 1 | 22.31 | 0.494 | 4 | 190.9 | NOT-ADMITTED |
| w08 | 1 | 25.14 | 0.547 | 0 | 141.4 | NOT-ADMITTED |
| p2_b1_01 | 2 | 41.44 | 0.631 | 3 | 997.5 | NOT-ADMITTED |
| p2_b1_02 | 2 | 49.05 | 0.571 | 3 | 982.9 | NOT-ADMITTED |
| p2_b2_01 | 2 | 56.74 | 0.694 | 6 | 1970.8 | NOT-ADMITTED |
| p2_b2_02 | 2 | 59.11 | 0.564 | 4 | 792.0 | NOT-ADMITTED |
| p2_b4_01 | 2 | 55.36 | 0.640 | 2 | 1286.0 | NOT-ADMITTED |
| p2_b4_02 | 2 | 49.09 | 0.575 | 2 | 1419.3 | NOT-ADMITTED |
| w01r | 1 | 45.98 | 0.671 | 9 | 1044.0 | NOT-ADMITTED |
| w04r | 1 | 44.22 | 0.615 | 5 | 800.3 | NOT-ADMITTED |

## Repeatability checks

- w01: 53.4 us -> repeat w01r: 1044.0 us — **INCONCLUSIVE-BAND-CHANGED** (loadavg 19.76 -> 45.98; band changed, rule (ii) not certifiable)
- w04: 123.6 us -> repeat w04r: 800.3 us — **INCONCLUSIVE-BAND-CHANGED** (loadavg 20.24 -> 44.22; band changed, rule (ii) not certifiable)

## Mechanism analysis

- Hypothesis (ctx-switch count predicts spread): **REFUTED**. w03: 1 switch, 1733.8 us; w08: 0 switches, 141.4 us; w02: 8 switches, 1986.6 us.
- Survivor (cgroup throttle delta vs spread): representative 12s sample = 120 periods, 0 throttled. This process's own quota is not throttling during sustained work (cumulative 64/3578 = 1.8%%). Spread is more consistent with neighbour contention / scheduler noise on shared cores.

## Verdict

**NO-ADMISSIBLE-OPERATING-POINT** — NO-ADMISSIBLE-OPERATING-POINT AT A 2-CPU QUOTA WITH THIS NEIGHBOUR LOAD: minimum observed spread is 53.4 us (window w01), 17.2 us (1.48x) above the 36.2 us requirement. No window satisfies rule (i). Whether a larger quota admits is UNTESTED here (see gpu-sim-vs2). The spread is unstable across repeats (53.4 -> 1044 us) and rule (ii) could not be certified (band changed), so even the best single sample cannot be treated as an admission.


## Environment finding

The measuring process runs in a scope cgroup with a 2-CPU quota (cpu.max=200000/100000). Cumulative scope: nr_periods=3578, nr_throttled=64 (1.8%%), throttled_usec=11.8s. Representative 12s sustained-workload sample: 120 periods, 0 throttled. The PI amendment's premise (no quota) holds at the app.slice level but NOT at the process scope; the scope IS quota-limited. Throttling of this process is intermittent (1.8%% of periods), not the 98%% seen in other live scopes. Recorded as a finding; the PI's original 'no quota' claim was wrong for the process scope.

## Per-window raw logs

experiments/gpu-sim-tfq/raw/<window_id>.log (one us value per line).
