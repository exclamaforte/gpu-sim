# gpu-sim-u2o.13.2 REPORT: spec schema shipped, training gated

CPU-only bead, no GPU.

## Shipped (real, tested)

epic_u2o/gpu_spec.py: 6-dim conditioning vector from sourced hw_5090
constants (sm 170, peak 82.6 flagged unsourced, mem 1792 GB/s vendor,
L2 96MiB, smem/SM, clock). Missing/unknown raises -- no silent zeros.
Unknown GPU raises KeyError. Smoke + 4 tests green.

## Gated (test-pinned)

Second-GPU measured rows don't exist; only rtx5090 has a spec. Training
with the vector waits for that data (hold-one-GPU-out is the test).
Design banked in deferred.md.

## Backing doc

deferred.md: only the 13.2 placeholder replaced.
