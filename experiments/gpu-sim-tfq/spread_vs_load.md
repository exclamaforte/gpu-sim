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

- w01: 53.4 us -> repeat w01r: 1044.0 us (fluky=False)
- w04: 123.6 us -> repeat w04r: 800.3 us (fluky=False)

## Verdict

**NO-ADMISSIBLE-OPERATING-POINT** — minimum observed spread 53.4 us (window w01), 17.2 us (1.48x) above the 36.2 us requirement.

## Environment finding

The measuring process runs in a scope cgroup with a 2-CPU quota (cpu.max=200000/100000) and IS throttled (nr_throttled=64, throttled_usec=11.8s). The PI amendment's premise that no quota exists holds at the app.slice level but not at the process scope. Recorded as a finding per the amendment's instruction.

## Per-window raw logs

experiments/gpu-sim-tfq/raw/<window_id>.log (one us value per line).
