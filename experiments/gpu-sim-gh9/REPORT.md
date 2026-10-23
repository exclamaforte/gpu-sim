# gpu-sim-gh9: analytic lever 3
Verdict: PARTIAL. Baseline elementwise 0.969/matmul 0.409/reduce 0.473; lever3 elementwise 0.181/matmul 0.179/reduce 0.486. Elementwise and matmul pass 0.25; reduce fails.
Floor: same-trace launch_bound case (1.0016us), not 6.6.1 host floor.
Noise floor: elementwise repeat spread median 24.89%, max 35.08%; 0.25 bar sits on that floor. High-spread cases: elementwise [1024] 35.08%, [65536] 34.78%.
Elementwise by band: <=65536 0.136, 1048576 0.212, 16777216 0.650. Residual is regime structure (cache-degraded large), not a constant offset; not fixable with available constants.
Calibrated on knee_cases (elementwise only); scored on fixture; no overlap. See metrics.json.
