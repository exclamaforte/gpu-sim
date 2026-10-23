# gpu-sim-v4e: validity envelope

Question: how far outside the calibrated grid does the byte-keyed model stay within 25%?

Method: 3 sizes below smallest calibrated size and 3 above largest, one op per class (relu, matmul, sum), measured once each on RTX 5090 with m93 harness. Scored against m93 calibration table (no refit). Clamp targets recorded.

## Results

### Below grid (clamped to smallest calibrated size)
| class | size | octaves below | rel err | within 25%? |
|-------|------|---------------|---------|-------------|
| elementwise relu | 1024 | 1.0 | +0.016 | yes |
| elementwise relu | 512 | 2.0 | -0.164 | yes |
| elementwise relu | 256 | 3.0 | -0.133 | yes |
| reduce sum | 2048 | 1.0 | +0.118 | yes |
| reduce sum | 1024 | 2.0 | +0.212 | yes |
| reduce sum | 512 | 3.0 | +0.236 | yes |
| matmul | 128 | 1.0 | +0.207 | yes |
| matmul | 64 | 2.0 | +0.504 | NO |
| matmul | 32 | 3.0 | +0.418 | NO |

Envelope below: elementwise and reduce hold for at least 3 octaves. Matmul holds only 1 octave below; fails at 2.

### Above grid (clamped to largest calibrated size)
| class | size | octaves above | rel err | within 25%? |
|-------|------|---------------|---------|-------------|
| elementwise relu | 536870912 | 1.0 | -0.514 | NO |
| elementwise relu | 1073741824 | 2.0 | -0.750 | NO |
| elementwise relu | 2147483648 | 3.0 | -0.875 | NO |
| reduce sum | 134217728 | 1.0 | -0.456 | NO |
| reduce sum | 268435456 | 2.0 | -0.737 | NO |
| reduce sum | 536870912 | 3.0 | -0.866 | NO |
| matmul | 16384 | 1.0 | -0.590 | NO |
| matmul | 24576 | ~1.5 | -0.885 | NO |
| matmul | 32768 | ~2.0 | -0.953 | NO |

Envelope above: NO class holds even 1 octave. Clamp returns a confident wrong number (45-88% error).

## Conclusion
The model is valid ~3 octaves below the grid for elementwise/reduce, 1 octave for matmul. Above the grid it fails immediately (first octave) for all classes. The clamp policy is unsafe in the extrapolated direction. Recommendation: document the envelope; do not silently clamp above the grid.
