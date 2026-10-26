# gpu-sim-hz3u: activation recomputation in the scale-out model

## Hypothesis (before running)

The scale-out model has no way to price activation checkpointing, so 405B comes out
INFEASIBLE at 80 GPUs and needs pp=100 at 800. Real 405B training uses full checkpointing
with pp=8-16. Adding a recompute knob should (a) leave every published number bit-identical
at the default, and (b) answer whether 405B@80 becomes feasible with full recompute and
what config wins at 800/8000.

## What was already there (deviation from the bead text, deliberate)

The bead asks for a NEW `hw['recompute'] = 0.0..1.0` float knob. The model already HAS a
recompute knob -- `hw['recompute']` in {"none","selective","full"} (gpu-sim-ooj, 2026-10-24)
-- with tests (tests/test_attention_flops.py), memory-side behavior (full keeps one layer's
activations) and a compute charge. A second parallel knob would be two APIs for one thing;
this bead reuses the existing one and wires it into the scale-out script. "Both settings"
in the acceptance = none vs full. Reported to the PI via `lab send`.

## Defect found while wiring (fixed here)

The existing full charge added the WHOLE 6P+attn compute a second time (+100%). Full
checkpointing replays one forward pass per step: 2P of the 6P matmul total, 1/3 of the
12*S^2*d attention total (bwd is 2x fwd) -- ~+33%, the standard figure (Chen et al. 2016;
Megatron-LM/DeepSpeed checkpointing docs ~30%). Corrected in
epic_cluster/parallelism_search.py; nothing published used recompute != none, so no
published number moves (verified: default matrix regenerates bit-identical, parsed-JSON
comparison against experiments/gpu-sim-scaleout/answers.json, diffs: 0).

## Outcome (negative result on feasibility, positive on the question)

- 405B at 80 GPUs stays INFEASIBLE with full recompute (all 4 cells: both conventions x
  batch 1/8). Mechanism: the most memory-frugal layout in the whole scored space needs
  99.5 GB with none and 93.5 GB with full, against 80 GB HBM -- and the
  weights/master/optimizer floor alone is 467.2e9 params x 16 B / 80 = 93.4 GB. Activations
  were never the blocker; no recompute setting fits 405B on 80 H100s under this accounting.
- At 800 the winner is unchanged (pp=100): 0.154 s ring (5.97%) / 0.201 s default
  (2.52%, 75.7 GB) vs 0.138/0.185 s without -- roughly a tenth more step time for 0.4 GB
  less memory. At 8000 also unchanged: (tp=8 pp=40 cp=25) ring 0.066 s (2.53%),
  (tp=8 pp=50 cp=20) default 0.096 s (0.13%).
- Bead item 4 (rewrite if feasible) did NOT trigger; the doc sentence now states the
  setting each number assumes (recompute=none) and carries the with-full numbers above.

## Reproduce

Script commit: b893b22 (scaleout_answers.py --recompute/--models flags new in this bead;
model fix in epic_cluster/parallelism_search.py on branch lab/gpu-sim-hz3u).

    python3 scripts/scaleout_answers.py --nodes 10,100 --json   # default; diff vs answers.json = 0
    python3 scripts/scaleout_answers.py --nodes 10,100,1000 --models llama3-405b --recompute full --json  # -> so_full_405b.json
    python3 scripts/scaleout_answers.py --nodes 10,100,1000 --models llama3-405b --recompute full         # -> answers_full_recompute_405b.md

## Artifacts in this dir

- so_full_405b.json / answers_full_recompute_405b.md: the 405B x {80,800,8000} matrix
  with full recompute (both conventions, batch 1/8).
- probe_405b_none_vs_full.txt: none-vs-full winner comparison that sized the result.
- memfloor_80.txt: min-over-all-scored-layouts memory (99.5 none / 93.5 full) + weights
  floor derivation (93.4 GB).

## Negative control (gpu-sim-xy8)

New assertion tests/test_recompute_charge.py::test_full_recompute_charges_one_forward_pass
pins the corrected charge. Demonstrated failing run: with the fix reverted in scratch
(old +100% charge restored by string replacement), the test FAILS
(`1 failed, 1 passed in 0.02s`, AssertionError at the exact-charge assert); with the fix
restored, `2 passed in 0.02s`. The memory-side companion test passes under both (it pins
the unmodified memory side, not the fix).
