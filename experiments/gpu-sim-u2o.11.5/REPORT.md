# gpu-sim-u2o.11.5 REPORT: writeup with sourced numbers

CPU-only bead. results_writeup.md rewritten: synthetic fixtures retired,
every figure from a MERGED artifact, methods as-run, negatives as the
deliverable (GBT wins, heavy tail, RMSLE degeneration, cross-attn loss,
superseded arms, dead ends kept).

## Anti-drift construction

gen_writeup_115.py emits the results block from metrics files;
tests/test_writeup_115.py asserts every emitted line verbatim in the
writeup + asserts no synthetic-fixture language remains. Pending-review
work (11.1/11.2/11.6/12.1) is cited by branch, never numerated -- numbers
land when the branches land, via the same generator pattern.

## Backing doc

experiment.md: only the 11.5 placeholder replaced. results_writeup.md is
the bead's main artifact (epic_u2o/, not experiments/).
