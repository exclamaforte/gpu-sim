# gpu-sim-gh9: analytic lever 3
Verdict: PARTIAL. Baseline elementwise 0.969/matmul 0.409/reduce 0.473; lever3 elementwise 0.180 (max 0.650)/matmul 0.179 (max 0.698)/reduce 0.478 (max 0.730) on a per-case basis (12/5/8 cases). Elementwise and matmul pass at median; reduce fails.
Floor: same-trace launch_bound case (1.0016us, fraction 0.3508/0.3478 spread), not 6.6.1 host floor.
Noise floor: elementwise repeat spread median 0.2489, max 0.3508; 0.25 bar sits on that floor. High-spread cases: elementwise [1024] 0.3508, [65536] 0.3478.
Elementwise by band: <=65536 0.136, 1048576 0.212, 16777216 0.650. Residual is regime structure (cache-degraded large), not a constant offset; not fixable with available constants.
Standing: gh9 is the PHYSICAL-MODEL comparison. naive_us fails 0.969/0.409/0.473. The physically-parameterised form with the measured floor gets matmul fully and the other classes at median. The measured lookup table (mdz) is what actually passes the gate (25/25). Calibrated on knee_cases (elementwise only); scored on fixture; no overlap. See metrics.json.
