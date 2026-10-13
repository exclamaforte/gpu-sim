# gpu-sim-u2o.17.4 REPORT: parity gate FAILs with numbers

CPU-only bead (fixture + supplement committed; no GPU).

## Criterion (pre-set, PI amendment)

Median |sim-measured|/measured <= 25% per op class + same dominant
class by total time. Set BEFORE the knee dose existed.

## Result: FAIL

| class | median err | bound | rank |
|---|---|---|---|
| elementwise (7 sizes) | 0.969 | FAIL | -- |
| matmul (5 sizes) | 0.409 | FAIL | dominates both |
| reduce (4 sizes) | 0.473 | FAIL | -- |

Rank-order preserved (matmul dominates measured and predicted), but every
class misses 25% by 1.6-3.9x. Leave-one-size-out is extrapolation-heavy
(edge sizes); the scaling laws interpolate better than they extrapolate.
This is a NEGATIVE result with numbers, not a hidden one: the sim path
beats naive everywhere yet is nowhere near parity. Gate stays closed.

## What would move it

Interior-size interpolation error (vs edge extrapolation), two-regime
elementwise form, and ultimately the trained predictor the bead
originally awaited. Each is new-leaf work with a pre-stated bound.
