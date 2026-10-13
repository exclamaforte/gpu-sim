# gpu-sim-u2o.9.1 REPORT: transformer baseline (Stage 1+2)

Ruling lapsed on numbers (measurable 39.4% >= 35%; mm_full 538 >= ~400),
recorded on bead. Dual-head encoder + sigma-floor NLL, 38,950,658 params
(d512/8h/12L/dff2048 + 33-dim numeric concat fusion + dual heads),
20 epochs / batch 32 / AdamW 3e-4 cosine / seed 0. Full job 6ae4df78 exit 0
(trial did all 200 steps in 34s; full job resumed 0 steps, re-evaluated).

## Three-way comparison (same split, same classes, log-space)

family_holdout_v1: test = relu 57 rows, val = mul 57, train = 342.
9.4 + 9.5 figures LOADED (9.5 from branch lab/gpu-sim-u2o.9.5, pending
review at write time -- provenance stated).

| triple | roofline 9.4 | GBT 9.5 | transformer 9.1 (this) |
|---|---|---|---|
| latency->latency | 0.5638 | 0.107 | 0.144 (bias +0.075) |
| latency->memory | 0.6873 | 0.027 | 0.260 (bias -0.147) |
| memory->memory | 1.1209 | (not in test) | (not in test) |

Headline: the transformer beats the roofline ~4x/2.6x and LOSES to the
numeric-only GBT (1.3x on latency, ~10x on memory). The sequence channel
earns nothing beyond the 33 numeric features on this holdout -- the
comparative question 9.5 was filed for, answered against the transformer
at this scale. val (mul) tracks train, so this is not a training failure;
it is a generalisation gap on unseen relu.

## Two bugs caught by the pipeline, both recorded not hidden

1. **Constant-feature detonation (normalisation, not architecture).** 16/33
   numeric features have train std exactly 0; z-scoring with (0+1e-9)
   mapped test-time differing constants to ~1e9 and the model to 1e14
   test error. Fixed with a train-fitted mask (constants -> 0, recorded
   in num_norm). No retraining: input-identity on train/val proven
   (max diff 3e-9, asserted in reeval_91.py); test re-evaluated from the
   same ckpt on CPU. Pre-fix numbers superseded, mechanism stated.
2. **Digest-load path doubled experiments/ prefix** (parents[1] vs [2]).
   First trial errored; fixed; CPU dry-run verified split + progress.

## Acceptance mapping

1. Negative control: tiny dual-head on add-launch_bound only overfits
   (train 0.115) and the eval flags collapse (heldout 1.00, disjoint
   regimes, collapsed=true). negctrl.json + green run. (Noted along the
   way: add->mul generalises, te~=tr -- sibling families are not collapse;
   the control uses regime collapse, honestly.)
2/3. Provenance enumerated by test (dataset keys exactly the 6 model
   inputs; kernel_hash a str ID, never a tensor; op_type absent; no
   hash-derived values) -- not by loader inspection.
4. NLL round-trips (mean,var)->loss on a synthetic cell to 1e-6; the
   sigma floor is proven to cap the loss as var_pred->0. DualHead naming
   quirk documented (softplus => variance, not log-variance; 9.3 owns it).
5. model_bias carried beside rows in 6.6, never a feature (asserted).
6. Per-triple abs + bias above, 9.4's classes.
7. Baselines loaded, paths recorded (metrics.json baselines_loaded).
8. Stage 1+2 stated (ruling lapsed); in-sample numbers in run history,
   holdout numbers labelled as such.
9. 4 known-OOB frozen rows named (hashes in metrics.json oob_rows);
   overlap with join 0; all 456 training sources pass the OOB guard
   (build aborts otherwise) -- in-spec VERIFIED, not asserted.
10. 38,950,658 params; ~0.15 s/step (200 steps / 34 s wall); peak
    16.4 GB; mean util 87.8% (trial 46426b05 telemetry).

## Reproduction

CPU: build_data_91.py (regen ~2 min) + negctrl_91.py (~5 min) +
tests/test_trans_baseline.py. GPU (documented, not replayed):
run_trans.sh -- 200 steps, ~40 s. The 157 MB ckpt is NOT committed;
run_metrics/metrics.json + reeval_91.py are the evidence. run.sh asserts
the CPU replay byte-identical.
