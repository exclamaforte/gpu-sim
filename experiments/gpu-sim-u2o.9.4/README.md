# gpu-sim-u2o.9.4 — the analytic roofline baseline, on real kernels

The deliverable is REPORT.md in this directory: the number the learned
model in 9.1 must beat, per regime, with both residuals carried (not
fitted) and the cancellation warning stated.

Headline: overall mean abs error 7.7238 -> 0.6581 (11.7x) by removing four
wrong assumptions, none of it by fitting -- but the overall signed bias of
-0.0245 is CANCELLATION (latency rows underpredict, memory rows
overpredict), not accuracy. 9.1 is judged per regime.

`report_verify.py` checks every checkable claim against committed paths.
`bash run.sh` re-runs it and byte-compares `metrics.json`. CPU-only.

(An earlier September pass at this bead fit ceilings on the 6.6.2 anchor;
its framework files were removed when the authored design reframed 9.4 as
the corpus report. History is in git.)
