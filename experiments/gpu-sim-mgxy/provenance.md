# gpu-sim-mgxy: how can a kernel measure faster than the empty kernel

Successor to 7m60 (flagging) and q9zk (reproduction). Those made the bad row
visible; this bead asks *why* the sign is reproducible in one cell.

## Measurement: the NULL cell (the decisive experiment)

`experiments/gpu-sim-mgxy/instrument_null.py` (log: `null_cell.log`, 20 repeats,
RTX 5090, prebuilt 8zqr binaries `empty_16_1_1_64_1_1.out` and
`k_divergent_branch_0000.out`). Per repeat it runs

    E  = empty binary, 20 trials        (the floor)
    E2 = the SAME empty binary again    ("scale" series of the null cell)
    S  = a real tiny kernel binary

and computes kernel_us for (E2,E) and (S,E).

    NULL cell, median estimator   : n=20 median=0.040us min=-1.328 max=1.616 negative=10/20 (50%)
    NULL cell, warm-min estimator : n=20 median=0.352us min=-0.160 max=1.824 negative= 1/20 ( 5%)
    REAL cell, median estimator   : n=20 median=0.624us min=-0.544 max=1.296 negative= 4/20 (20%)
    REAL cell, warm-min estimator : n=20 median=1.088us min=0.512 max=1.440 negative= 0/20 ( 0%)

**A kernel cannot be faster than itself.** The null cell is negative in half the
repeats under the shipped estimator, with magnitudes up to -1.33 us, so a
negative `kernel_us` is the estimator's own noise and carries no information
about the kernel. That rules out "the kernel is genuinely faster" and explains
why the sign reproduces in the same cell (that cell's floor estimate happens to
land high) while the magnitude varies run to run.

Within a single 20-trial run the empty series spans 1.2-6.8 us (see the
`empty_trials` field per repeat in the log), so the median of 20 cold launches
is not a stable floor.

## Fix: a warm, trimmed, MINIMUM floor

    floor_ms(empty) = min(empty[WARM_LAUNCHES:])          # WARM_LAUNCHES = 5
    kernel_us       = (median(warm(scale)) - floor_ms) * 1000

Cost only ADDS to a launch, so the floor of a pure-launch kernel is its fastest
well-conditioned launch; a median also carries the cold and jittered samples.
Both series are trimmed the same way so they describe the same launch regime.

Row now records `estimator: "warm_min_floor"`, `empty_floor_ms`,
`floor_jitter_us`, and `within_floor_noise` (true when a negative is smaller
than the floor's own warm spread — i.e. indistinguishable from the noise).

NOT DONE, deliberately: `kernel_us` is never clamped to 0 (7m60's flagging stays
the safety net, and clamping would hide hypotheses 1-3), and the admission
threshold is unchanged, so this bead does not move the arm's cell count.

## Evidence the fix does not buy its result with a false negative

The same run shows the real tiny cell going 4/20 negative -> 0/20 negative while
`test_above_threshold_still_usable` keeps a genuine 195 us kernel usable.

## Row

- Test: `tests/test_mgxy_warm_floor.py` (5 tests)
- Bound: `test_floor_ignores_cold_launches`
- Mutation: `mgxy_median_floor` (put the cold median back) -> 1 failed / 4 passed,
  failing node is the bound test; restored -> 5 passed.
