# gpu-sim-mdz: calibration sweep (final, per-op)
86 calibration cases (incl. quarter-octave), 3 invocations (90 GPU-min), disjoint from scored fixture.
Result 2 (per-op inference): elementwise median 0.025 (max 0.104), matmul 0.038 (max 0.175), reduce 0.024 (max 0.196). All 25 scored cases pass 0.25.
What passes is a MEASURED LOOKUP TABLE WITH LOG-LOG INTERPOLATION, not a physical law (naive_us fails 0.969/0.409/0.473).
Calibration density cost: ~86 measured cases to grade 25 scored cases (~3.4x), concentrated at <=1.19x spacing in the L2->HBM transition.
See metrics.json.
