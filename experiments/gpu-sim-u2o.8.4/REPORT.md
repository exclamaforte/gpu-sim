# gpu-sim-u2o.8.4 REPORT: representation ablation ends undecidable (path B)

PI ruling 2026-10-13 pre-committed (B): if capacity+log still vacuous,
report `undecidable at this scale` with curves and close honestly.
This is that report. D_REPR stays open.

## What was run

Two matched-size scales, identical splits/seeds/hparams except encoder dims:

| scale | dims | params | epochs | train rows | job |
|---|---|---|---|---|---|
| 199K | d64/4h/2L/dff256 | 199169 | 100 | 2494 | f34f4d02 exit 0 |
| big | d256/8h/6L/dff1024 | 4936449 | 100 | 2494 | d0c8952d exit 0 |

Same train rows (random-split train, seed 42), same seed 0, same
AdamW 3e-4/cosine/batch 64, log(ms) targets (Gate 0: `gate0_TARGET.md` --
arms never trained raw). Per-scale dirs isolated
(`runs/gpu-sim-u2o.8.4`, `runs/gpu-sim-u2o.8.4-big`); `metrics_199k.json`
/ `metrics_big.json` hold the per-arm per-suite MAPEs; `curves.json` the
loss trajectories (every 4th train step + all val points).

## Numbers

Target variance `var(ln ms) = 6.209` (mean-predictor MSE floor);
mean-predictor MAPE on ms = 3.11. Targets span 5 decades
(0.0017..330.5 ms, median 0.004).

| arm | scale | train MSE (min) | val MSE | train MAPE | suite MAPE range |
|---|---|---|---|---|---|
| source | 199K | 2.53 | 3.65 | 3.06 | 0.93..2.94 |
| ptx | 199K | 2.51 | 6.52 | 3.44 | 0.93..7.33 |
| hybrid | 199K | 2.45 | 5.59 | 3.36 | 0.94..3.41 |
| source | big | 2.49 | 3.88 | 3.26 | 0.94..7.35 |
| ptx | big | 2.52 | 6.50 | 3.10 | 0.89..2.85 |
| hybrid | big | 2.52 | 5.55 | 3.08 | 0.94..2.93 |

The scale-invariance is the finding: **25x params moves the best train floor
2.45 -> 2.49 (unchanged within noise) and val barely at all.** Capacity is not binding; the
learnable signal at this data scale is exhausted (train MAPE ~310% ~=
predicting the mean). Hybrid_red is negative on all 4 suites at both
scales (0/4 criterion 1).

## Why no verdict is reported

Both `metrics_*.json` files compute `verdict: single-modality-sufficient`
mechanically (a single arm is trivially "within 2%" of a worse hybrid).
That verdict is VACUOUS: with all arms at mean-predictor level, the
>5%/2% criteria compare noise, and per-suite MAPE over 5 decades is
dominated by a few outlier rows (e.g. random-suite MAPE 7.35 vs 0.89 for
arms whose val-MSE differs 2x the other way). Reporting source-only off
this would bless a representation on noise. The criteria assumed learning;
learning did not happen.

One replicated hint, stated as hint only: val-MSE ordering
source < hybrid < ptx holds at BOTH scales (3.65<5.59<6.52;
3.88<5.55<6.50). Consistent with source carrying the most signal, but at
mean-level absolute error it supports no decision.

## Conclusion (B)

Representation comparison is undecidable at ablation scale (2.5K rows,
<=5M params, token-only inputs, absolute-ms target over 5 decades).
Closing the ablation honestly; D_REPR stays open until a decisive
comparison exists (likely needs orders more data and/or a different
target, e.g. within-family ranking -- out of scope here).

## Reproduction

`run_arms.sh` (default OUT) reproduces the 199K scale;
`OUT_DIR=... MODEL_ARGS="--d-model 256 --n-heads 8 --n-layers 6 --d-ff 1024"`
reproduces big. Two GPU jobs (~17 min + ~15 min observed). NOT re-executed
post-freeze: rerunning burns two GPU jobs to reproduce a null whose
evidence (metrics + curves, extracted from the job outputs) is committed
here. `compare_84.py <run dir>` regenerates either metrics file from
per-arm `arm_metrics.json`.
