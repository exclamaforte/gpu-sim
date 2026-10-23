# gpu-sim-gh9: analytic lever 3
Verdict: PARTIAL. Baseline elementwise 0.969/matmul 0.409/reduce 0.473; lever3 elementwise 0.484/matmul 0.179/reduce 0.436. Matmul passes 0.25; elementwise/reduce do not. Calibrated on knee_cases (elementwise only); scored on fixture; no overlap. Noise floor elementwise median spread 0.249, max 0.351. See metrics.json.
