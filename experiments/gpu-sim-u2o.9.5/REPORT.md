# gpu-sim-u2o.9.5 REPORT: GBT baseline over hand-engineered features

Stage 1 + Stage 2 (7.7 is CLOSED/merged, so the staged gate is satisfied).
CPU only. Model: sklearn HistGradientBoostingRegressor (the design names
XGBoost/LightGBM; neither is installed -- same model class, substitution
recorded in metrics.json). Hparams: max_iter 300, max_depth 6, lr 0.05,
min_samples_leaf 5, l2 0, seed 0.

## Join (acceptance 1)

8.3 features.jsonl x 6.6 targets.jsonl on kernel_hash: 456/456, shortfall
none. Median agreement asserted per row (features.measured_median_ms ==
targets.median_runtime_ms to 1e-12). Join-field reconciliation is the 9.5
scope: the 33 vector internals are 8.3's tested claim
(test_feature_vector.py), cited not re-derived; the PER-KERNEL byte
convention is printed beside every figure (9.4's committed
bandwidth_convention string, loaded).

## Transform (acceptance 2)

y = ln(ms + 1e-6), pred_ms = exp(pred). Round-trip on 0.002536 bounded by
the guard (2e-6): the +1e-6 is a KNOWN bounded asymmetry, stated, not a
silent one.

## model_bias (acceptance 3)

6.6's model_bias is carried BESIDE each row for audit and never enters the
feature matrix. Asserted: no bias-substring key in any feature row, vector
length 33 on all 456 rows (test_no_bias_field_in_features).

## Harness (acceptance 4)

Per-regime-triple mean abs err + signed bias in log space -- 9.4's units
(9.4 REPORT table: 0.5638 / 0.6873 / 1.1209). Negative control: fit on
launch_bound only, eval measurable; the census flag fires
(REGIME-IMBALANCED, total-variation 0.78 > 0.25) and is asserted to fire.

## Stage 1: in-sample (acceptance 5)

Fit on all 456, evaluated on all 456, labelled IN-SAMPLE with the regime
census beside it (launch_bound 283 / measurable 173; triples 220/204/32).
Triple errors 0.051/0.019/0.012. NO generalisation claim is made for these
numbers; they exist to show the harness runs and the model fits.

## Stage 2: family_holdout_v1 (thin holdout, stated)

7.7's every-5th digest rule on the 8 stage-1 digest groups: test = relu
(57 rows), val = mul (57), train = 342. Zero shared digest groups across
sides; all guards raised-or-passed. Test triple errors: latency->latency
0.107 (n=29, bias -0.018), latency->memory 0.027 (n=28, bias -0.001);
memory->memory absent from this test template (in val, n small).

## Three-way frame (acceptance 6)

9.4's figures LOADED from the committed artifact (per_regime_achieved,
residual_A/B, families, convention -- equality-asserted in tests), never
recomputed:

| triple | roofline abs (9.4) | GBT holdout abs (this) |
|---|---|---|
| latency->latency | 0.5638 | 0.107 |
| latency->memory | 0.6873 | 0.027 |
| memory->memory | 1.1209 | (not in test template) |

The GBT beats the roofline 5-25x on the held-out relu template with near-zero
signed bias. That delta is attributable to the feature channel over the
roofline's arithmetic -- the sequence channel (transformer, 9.1) slots into
the same frame later. Per-family figures carry per-kernel counts (all
families single-kernel, n=57 rows each -- never presented as multi-kernel
family claims).

Caveat stated, not hidden: the holdout is ONE template (57 rows). The number
is real but narrow; broadening the holdout needs the 40-group merged split
with feature vectors beyond stage 1, which do not exist yet.
