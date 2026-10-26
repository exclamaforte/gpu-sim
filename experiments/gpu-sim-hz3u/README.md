# gpu-sim-hz3u: activation recomputation in the scale-out model

## What the bead asked, what the host measured, what the model does

The bead asked for a float knob (0.0..1.0), default 0.0, bit-identical published numbers, and a
re-run of the 405B matrix with full recompute. While the first implementation was in flight, the
host measured the real thing on the lab 5090 (commits 87e0e69, a074e57, 2d968bc,
experiments/gpu-sim-recompute/): full per-block checkpointing costs **1.24-1.27x step time** and
retains **0.33 of peak at seq 8192 / 0.49 at seq 4096** (same width). Those measurements -- not a
rule of thumb -- are what the model now charges:

- TIME: compute term x (1 + 0.25*recompute). The 1.33x forward-FLOP rule the first attempt used
  is slightly pessimistic; the rig says 1.25x.
- MEMORY: stored activations x (1 - 0.67*recompute). 0.67 is the seq-8192 value (top of the
  measured 0.5..0.67 range); it overstates the saving at shorter sequence, stated here and in
  the code so short-seq recompute fits get their own calibration. A seq-dependent fit on two
  clean points was rejected as overfit.
- A 0.55 fallback was tried first and REJECTED by the bead's own falsifiable target: with 0.55
  the 70B/batch-32/80-GPU cell picks pp=2 at 2.25% margin instead of the measured pure-FSDP
  flip, because 0.55 keeps 0.45 where the rig keeps 0.33 and the FSDP layout needs 90.1 GB.
  The 0.55 run is recorded here so the next person does not retry it.

## Outcome

- ACCEPTANCE CELL (PI 17:50, reproduced exactly): 70B at 80 GPUs, batch=32. Without recompute
  the model reproduces the old row -- (tp=1 pp=4 cp=1 dp=20 z3) at 2.769 s (0.45%, 58.0 GB).
  With full recompute it flips to (tp=1 pp=1 cp=1 dp=80 z3) at 2.920 s (14.07%, 70.9 GB):
  no pipeline at all, pure FSDP, above the 2.474 s chargeless proxy by the 1.25x charge, far
  better separated than the tie it replaces. The margin widens proxy-10.6% -> model-14.07%
  because the charge lands unevenly on compute-heavy shapes; the config, the memory, and the
  direction are identical.
- 405B: full recompute changes NOTHING. 80 GPUs stays INFEASIBLE (optimizer state alone is
  70.9 GB -- 14 B/param on the nominal 405e9, fp32 master + fp32 Adam + bf16 grads -- and
  recomputation cannot close a gap that size); the 800/8000 winners do not move
  (still pp=100 at 800: 0.150 s ring / 0.197 s default; same configs at 8000: 0.065 / 0.096 s).
- Default (recompute=0.0) regenerates the committed scaleout matrix bit-identically
  (parsed-JSON diff 0), model_facts.json carries no changed value, gate 8/8.

## Reproduce

Model + script on branch lab/gpu-sim-hz3u (rebased onto 2d968bc):

    python3 scripts/scaleout_answers.py --nodes 10,100,1000 --json   # default; diff vs answers.json = 0
    python3 scripts/scaleout_answers.py --nodes 10,100,1000 --models llama3-405b --recompute 1.0 --json  # -> so_full_405b.json

The 70B/batch-32 acceptance numbers are computed by scripts/check_docs_model_numbers.py
(facts["recompute_70b_b32_80"]) and pinned by tests/test_docs_narrative.py, not by the script.

## Artifacts in this dir

- so_full_405b.json / answers_full_recompute_405b.md: the 405B x {80,800,8000} matrix
  with recompute=1.0 (both conventions, batch 1/8).
- README.md: this file.

## Negative controls (gpu-sim-xy8)

- tests/test_recompute_charge.py pins the knob contract (default-is-zero bit-identical,
  1.25x compute at 1.0, 0.67 activation retention recomputed from the model's own geometry,
  legacy strings mapped, garbage rejected). Demonstrated failing run: model file reverted to
  pre-knob state in scratch -> all 4 tests FAIL; restored -> 4 passed.
- The alpha choice is guarded the same way in reverse: with ALPHA=0.55 the acceptance cell
  yields (1,2,1,40,3) at 2.25% margin, failing the FSDP-flip assertion -- the constant is
  pinned by the phenomenon, not by taste.

## Lesson, verbatim (PI, second time today a correct measurement produced a wrong inference)

Check which term is binding before attributing an infeasibility to the term you just measured.
The first attempt blamed activations for 405B@80 and overcharged the fix; the binding term was
70.9 GB of optimizer state, and the knob's actual value is one FSDP flip in a batch-heavy 70B cell.
