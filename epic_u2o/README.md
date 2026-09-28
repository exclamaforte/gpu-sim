# `epic_u2o/` — what lives here, and what does not

The CUDA-runtime-prediction epic's code and specifications. One rule, and the
rest is detail:

> **Source and specifications live in `epic_u2o/`. Measurements and results
> live in `experiments/<bd-id>/`. Build artifacts and caches live nowhere in
> git.**

## Why the rule is enforced rather than assumed

`master` once tracked 2 of ~40 modules here, with the rest existing only as
untracked files in the working tree. That was not tidiness: it made three bead
merges collide, and it left **master unable to run its own preflight** —
`safe_harness.py` imports `preflight_validator` and `gpu_recovery` lazily, and
both were untracked. A fresh clone failed as
`Pre-flight FAILED: static scan error: No module named ...`, which reads like
a bad kernel rather than an incomplete checkout.

The full per-path classification, with the decision and a one-line reason for
each, is `experiments/gpu-sim-u2o.6.11/tracks_inventory.md`.

## What is deliberately untracked

* **Result artifacts** (e.g. `gate_u2o17_real_vs_naive_trace.json`). Results
  belong to the bead that produced them, in `experiments/<bd-id>/`, stamped
  with a version and the ts job id. A JSON sitting beside the source is
  unversioned by construction — which is how a synthetic fixture nearly
  became the record of a real measurement.
* **Placeholder specs** — the literal `# placeholder gpu-sim-u2o.X` stubs. They
  are tracked by decision *against*, in the inventory, not hidden by a rule, so
  the gap is visible instead of merely absent.

## Before you add a file here

1. Is it source or a specification? Both belong. Is it a *result*? It belongs
   in `experiments/<bd-id>/`.
2. Does it contain numbers you measured? It is a result, whatever its
   extension.
3. If you are fixing a tracked file, do not add a sibling "just in case" —
   untracked duplicates are what made the merges collide in the first place.
