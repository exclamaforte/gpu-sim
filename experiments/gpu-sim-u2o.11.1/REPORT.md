# gpu-sim-u2o.11.1 REPORT: Step-1 overfit sanity on real data

CPU-only bead, no GPU used (~4.3 min CPU).

## Result

32 fixed real rows overfit to 0.7% of initial NLL (22.79 -> 0.15, 150
full-batch steps, tiny d64/4h/1L dual-head). LR=0 control bit-exact.
Pipeline (tokenize -> batch -> model -> shared NLL) proven functional on
real measured rows.

## Two control-design errors caught during the run (both mine)

1. LR=0 control compared first-vs-last ACROSS shuffled mini-batches:
   different batches differ even with frozen weights (0.69 drift). Fixed
   with full-batch (same 32 rows every step).
2. Then dropout: train-mode forward noise moved the frozen loss by 0.017.
   Fixed with eval mode for the control run (no dropout => exact).
Lesson: a "no-change" control must fix the batch AND the sampling noise,
or it measures its own sloppiness.

## Cost note

Tiny model, but seq-512 x 32 rows x 170 steps ~ 4 min CPU. The first attempt (400 mini-batch steps x 2 runs) blew a
900 s timeout -- recorded so the next bead sizes CPU budgets correctly.

## Backing doc

experiment.md: only the 11.1 placeholder replaced; 11.2-11.6 untouched.
