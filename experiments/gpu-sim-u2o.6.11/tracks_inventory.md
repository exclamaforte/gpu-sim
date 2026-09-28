# gpu-sim-u2o.6.11 -- what `epic_u2o/` is, and what belongs in git

Every path under `epic_u2o/` in the working tree, classified. `master`
tracked 2 of these at the start of this bead (now 8, as earlier beads
merged their own fixes); the rest existed only as untracked files in the
main checkout, which is what made three bead merges collide and made one
look like it had lost its artifacts.

## The finding that decided the policy

Master was not merely untidy, it was **not self-consistent**. A fresh
clone of master cannot run the preflight at all: `safe_harness.py`
imports `epic_u2o.preflight_validator` and `epic_u2o.gpu_recovery`
*inside functions*, and both were untracked. Reproduced:

```
$ git clone /home/gabe/Documents/gpu-sim clonetest && cd clonetest
$ python3 -c "from epic_u2o.safe_harness import SafeHarness; \
             SafeHarness().preflight_check('__global__ void k(float* a){}',[1,1,1],[32,1,1],0)"
Pre-flight FAILED: static scan error: No module named 'epic_u2o.preflight_validator'
```

The dangerous part is how it fails: not an ImportError at import time, but
a **preflight validation error**, which reads like the kernel being wrong
rather than the checkout being incomplete. A reviewer or a future agent
would have gone looking for a CUDA bug.

## Classification

| path | state at bead start | category | decision | reason |
|---|---|---|---|---|
| `epic_u2o/batch_measure.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/bottleneck_check.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/compare_baselines.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/complete.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/corpus_miner.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/deferred.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/eval.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/experiment.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/experiment_program.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/features.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/future_extensions.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/gate_u2o17_capture_and_compare.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/gate_u2o17_real_vs_naive_trace.json` | UNTRACKED | result-artifact | **DO NOT TRACK** | a result artifact of a bead that is not closed; belongs in experiments/<bd-id>/ |
| `epic_u2o/generalization_suites.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/generator.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/generator_multistream.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/generator_stage1.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/generator_stage2.py` | tracked in master | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/generator_stage3.py` | tracked in master | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/generator_stage4.py` | tracked in master | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/generator_stage5.py` | tracked in master | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/gpu_control.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/gpu_recovery.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/harness.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/kernel_driver.py` | tracked in master | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/kernel_names.py` | tracked in master | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/measurement_spec.md` | UNTRACKED | spec | **TRACK** | written specification with real content |
| `epic_u2o/measurement_stats.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/metrics.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/metrics_eval.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/model.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/model_arch.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/model_specs.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/preflight.md` | UNTRACKED | spec | **TRACK** | written specification with real content |
| `epic_u2o/preflight_validator.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/ptx_tokenizer.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/queue_state.py` | tracked in master | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/ranking_metrics.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/recovery.md` | UNTRACKED | spec | **TRACK** | written specification with real content |
| `epic_u2o/repo_scaffold.md` | UNTRACKED | spec | **TRACK** | written specification with real content |
| `epic_u2o/representation_ablation.md` | UNTRACKED | spec | **TRACK** | written specification with real content |
| `epic_u2o/results_writeup.md` | UNTRACKED | result-artifact | **DO NOT TRACK** | a results document whose own text says its numbers are from synthetic fixtures; results belong in experiments/<bd-id>/ |
| `epic_u2o/roofline.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/safe_harness.py` | tracked in master | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/split.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/structural_split.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/throughput_tracker.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/tokenizer.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/toolchain.md` | UNTRACKED | spec | **TRACK** | written specification with real content |
| `epic_u2o/training_loop.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/training_loop.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/training_throughput.md` | UNTRACKED | spec (placeholder) | **DO NOT TRACK** | literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history |
| `epic_u2o/training_throughput_profile.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/tree_baseline.py` | UNTRACKED | source | **TRACK** | code the harness/generators import; without it master is not self-consistent |
| `epic_u2o/v1_scope.md` | UNTRACKED | spec | **TRACK** | written specification with real content |

**Tracked: 41 paths. Not tracked: 14 paths.**

## What is deliberately NOT tracked, and why

- `epic_u2o/complete.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/deferred.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/eval.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/experiment.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/gate_u2o17_real_vs_naive_trace.json` -- a result artifact of a bead that is not closed; belongs in experiments/<bd-id>/
- `epic_u2o/generator.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/harness.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/metrics.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/model_arch.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/model_specs.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/split.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/training_loop.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history
- `epic_u2o/training_throughput.md` -- literal '# placeholder' / one-line stub; the audit notes cite these as evidence of the fabrication history

Note in particular on two of them:

- `results_writeup.md` states in its own first paragraph that its numbers
  come from synthetic fixtures, and closes with "Known limits: no real-GPU
  timings yet". It is a results document, and results live in
  `experiments/<bd-id>/` where the owning bead can stamp them.
- `complete.md` is the single line `# Epic complete: all sub-epics closed`.
  It is the artifact the fabrication audit (gpu-sim-u2o.15) cites. Tracking it
  would put the record of a fabricated closure into master permanently.
- `results_writeup.md` is a result document whose own text says its numbers
  come from synthetic fixtures. Results belong to `experiments/<bd-id>/`,
  where the bead that owns them can stamp and version them.

## One thing found while verifying the admission set

Before committing, every file about to be tracked was grepped for the word
"placeholder" — the same check the inventory uses to keep the stub specs out.
Three hits, and two are benign docstrings about placeholder *numbers*. The
third is not:

`epic_u2o/preflight_validator.py:163-176`,
`PreflightValidator._run_under_fakegpu`, returns
`(True, "Preflight passed (placeholder)", "")` **unconditionally**. The
fake-GPU smoke test that method exists to perform does not happen, and reports
success. `safe_harness.preflight_check()` imports the static half of this
module, so this stub is reachable from the harness preflight — the same shape as
the 6.1.1 defect (a preflight that reports success without testing anything),
in the epic that was closed once already on evidence that included this file.

Filed separately rather than fixed here: this bead is a tracking decision, and
the fix is either implementing the path against the `-cudart=none` route
(gpu-sim-u2o.6.1.3) or making the stub RAISE instead of returning success so
"not implemented" can never be read as "passed".

Worth stating plainly: tracking this file is what makes the stub *reachable*.
A fresh clone of master today cannot import it at all, so the preflight fails
loudly with a missing module. Tracking is a net improvement — the failure
becomes visible rather than a silent pass — but it is not a fix, and the
honest reading of the change is "one visible failure replaced by one
invisible one until the stub is addressed".

## The rule, in one line

Source and specifications in `epic_u2o/`; **measurements and results in
`experiments/<bd-id>/`; build artifacts and caches nowhere in git.**
The full statement is `epic_u2o/README.md`.
