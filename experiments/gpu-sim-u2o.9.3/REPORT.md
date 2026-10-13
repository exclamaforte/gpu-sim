# gpu-sim-u2o.9.3 REPORT: noise-aware loss (shared, tested)

CPU-only bead: no GPU used. Shared loss semantics, not another training run.

## What was wrong

Two live inconsistencies around the dual head (9.1's documented naming
quirk, now resolved without renaming -- renaming `log_var_head` would
break committed checkpoint keys):

1. `GaussianNLLLoss` (model.py) takes TRUE log-variance but nothing feeds
   it that (DualHeadPredictor emits softplus'd VARIANCE). It is unused by
   every training path -- now documented as such, kept for import compat.
2. `TrainingLoop._gaussian_nll_loss` exp()'d the head's already-positive
   variance output (wrong optimum), in train_step (both AMP branches) and
   evaluate. 9.1/8.4 never trained through this loop, so no run is
   invalidated; the loop is 9.6's, the correction is 9.3's writ (flagged).

## What 9.3 ships

- `gaussian_nll_with_floor(mean, var_pred, target, sigma)` in model.py:
  NLL under N(mean, var_pred + sigma^2). Sigma is 6.6's sigma_log per
  cell, added beside the prediction (never inside it); 0.0 reduces to
  plain NLL. Contract: var_pred is a VARIANCE (the head's emission).
- Loop delegates to it, with `_batch_sigma` plumbing (sigma key when the
  batch carries it, else 0.0 for single-trial CUDADataset rows).
- `train_trans_91.nll_floor` is now a thin alias (single source; 9.1's
  numbers stand unchanged -- same math, verified equal to 1e-6).

## Evidence (tests/test_noise_loss.py, 8 tests, CPU)

- alias == shared on random tensors (1e-6)
- proper scoring: 4000 synthetic cells ~ N(0, 0.25); loss grid over
  candidate var {0.05: 1.039, 0.1: 0.117, 0.25: -0.186, 0.5: -0.093,
  1.0: 0.127} minimized exactly at truth 0.25
- floor: var 1e-9 + sigma 0.2 -> 1.52 (bounded); without sigma -> 1.25e8
  (collapse explodes) -- the floor is load-bearing
- negative control: log-variance numbers fed into the variance slot give a
  different loss (disagreement proven) -- the variance contract is not
  vacuous
- head emits strictly positive variance (DualHeadPredictor + create_model)
- loop delegates exactly + sigma defaults/passthrough

Suite: 484 passed; 7 failures are the pre-existing master set (unchanged).
Rebased onto master post-9.2-merge (486c348) cleanly. No GPU: queue untouched.
