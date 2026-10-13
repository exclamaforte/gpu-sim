# gpu-sim-u2o.11.4 REPORT: headline from 10.4's verdict

CPU-only bead, no GPU, no re-run (per PI ruling on this bead).

## The judged answer

NO. On latency->latency the transformer (0.144) beats the roofline
(0.564) ~4x and loses to GBT (0.107) ~1.3x. On latency->memory it beats
the roofline (0.687 vs 0.260) ~2.6x and loses to GBT (0.027) ~9.5x. The
sequence channel earns nothing over numeric features at this scale; the
head-to-head story is complete and negative, stated plainly.

## Anti-drift construction

Every ratio in the headline is computed from 10.4's cells at write time
(headline_114.py) and the test asserts headline == source cells +
ratios recompute. Prose cannot drift from numbers (the Sep-28
numerator-side lesson + 6.11 lint direction, applied).

## Axes without a headline (in the artifact, not dropped)

Ranking, calibration, generalization suites (10.3 frozen),
memory->memory learned cells. 11.2's failure-mode context is cited as
pending-review, not depended on.

## Backing doc

experiment.md: only the 11.4 placeholder replaced.
