# gpu-sim-bk9: kernel-work target and validity domain

## Supervised validity domain

Training is permitted only on cells where `target_usable` is true: the kernel's
own work must exceed the measurement noise floor by 5x. Across the five
families (elementwise, reduction, matmul, transpose, softmax-like) and five
sizes (2^10 .. 2^24), the usable cells are: elementwise 2^22; reduction 2^22;
matmul 2^18, 2^22, 2^24; transpose 2^22, 2^24; softmax-like 2^22. All other
(family, size) cells are excluded because `kernel_us` is below the gate
threshold or the row is grid-capped (grid.x capped at 65535, so the
2^24 elementwise/reduction/softmax runs only cover 16.7M of 16.8M elements).
The gate is `target_usable iff kernel_us >= 5 * max(within_process_std_us,
residual_661_us) and not grid_capped`; the 6.6.1 residual inputs (us) are
1024:0.018, 16384:0.068, 262144:0.078, 4194304:0.004, 16777216:0.027.
Reported training target is `kernel_us` (batched scale minus batched empty, K=40
launches per region); `reported_us` (single-mode median) is retained only as a
diagnostic and must never be used as a training label. `kernel_share` is
provided for provenance. gpu-sim-tsy cites this paragraph as the supervised
validity domain.