# TP payload scheme decision (gpu-sim-oez)

## Decision: (b) keep the model's Megatron-shaped payloads; label the scheme.

The search model (epic_cluster/parallelism_search.py) charges TP all-reduce as
L * 4 * (d * S_local * dtype) per rank per step (full-d row-parallel output).
The harness (tools/torch_parallel/train.py BlockTP) reduces a qkv-shaped tensor
of size batch * (S_local-1) * (3*d/tp) * dtype (measured: 1560576 B =
8 * 127 * 384 * 4).

## Why not (a) move the harness to Megatron placement

The model's weight layout (column-parallel qkv, row-parallel proj) already
follows Megatron's scheme. The all-reduce in _ColLinear.backward is on the
input-gradient gx (full d), and _ReduceSum (row) reduces the full-d output.
But the interposer's measured payload is 3*d/tp, which does not match either.
The discrepancy is in how the interposer attributes bytes (count field 260096
vs bytes field 1560576, a 6x ratio), not in the model. Resolving the
interposer's byte accounting would require changes to src/fakegpu_nccl.c that
are out of scope for this bead and would not change the ranking (sizes scale
uniformly within a scheme).

## Why (b)

- Counts are validated on hardware (16 tp AR / 24 cp sends per rank per step,
  exact). Counts are shape-independent, so the ranking's comm terms scale
  uniformly within a scheme.
- The ranking order depends on relative comm sizes, which are self-consistent
  within the Megatron scheme. Switching schemes scales all tp terms by a
  constant factor, preserving order.
- Absolute step times are already labelled "modelled only" (not hardware-
  validated) per the epic.

## What this validates / does not

- VALIDATES: counts (16, 24) and their shape-independence.
- DOES NOT VALIDATE: byte magnitudes or absolute ms. Those remain modelled-only
  until either (a) the interposer's byte accounting is reconciled with the
  tensor shapes, or a separate bead moves the harness to a single named scheme.

## Scheme label

All tables produced by parallelism_search are labelled "Megatron-shaped payload
model (full-d row-parallel output)". The harness byte audit (5o4.7) is labelled
"harness scheme (qkv-shaped, batch included)". The two must not be conflated.
