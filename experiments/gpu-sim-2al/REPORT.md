# gpu-sim-2al: stage 2 on powered target

Verdict: CONFIRMED. Primary protocol computed by primary_protocol.py from
predictions_primary.jsonl (408 rows = 3 models x 136 test rows; all finite).
Within-family size/config interpolation (pooled): roofline 1.0784, GBT 0.3709,
transformer 2.3137, mean predictor 2.7015. Transformer (2.3137) loses to GBT
(0.3709), confirming 11.4 on a sound target. The transformer cell moved from
the earlier 2.4205 to 2.3137 because the NaN fix used z-scored physical
features and a smaller model; GBT/roofline/mean are unchanged. Numeric-only
(no source tokens), 9/33 informative slots. The both-heads clause is
unsatisfiable (only latency->latency constructible). Secondary leave-one-family-out
is INVALID_UNINTERPRETABLE (unnormalized descriptors). Mean predictor is stated
for every cell. Supersedes gpu-sim-u2o.11.4. See experiments/gpu-sim-2al/metrics.json.
