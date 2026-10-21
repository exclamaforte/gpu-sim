# Variance measurement (preliminary)

Raw per-iteration durations in /tmp/scratchpad/iters_4k.log and iters_2m.log
(runs via forcenet, 4KB and 2MB payloads, GPUSIM_ITERS=20/5).

4KB (1024 floats): first iter 283575 us (init overhead), rest median 6830 us,
  min 6677, max 4641746 (outlier), spread ~4.6e9 us (dominated by one outlier).
2MB (524288 floats): first iter 164257 us (init), rest median 8524 us,
  min 6717, max 52702, spread 45985 us.

The 2MB spread (45985 us) exceeds the PI's 2MiB tolerance of ~20 us by ~2300x,
indicating the measurement is dominated by scheduler noise / outliers on this
oversubscribed box (load avg ~20-40). The futex wakeup is in place but the
box state makes the variance number unusable for the gate's payload selection.
A clean-box re-run is required (PI ruling). Box state recorded in 6kp artifact
(box_state.txt): nproc=2 cgroup quota, loadavg ~20-40, ~3274 tasks.

Per-piece fixed cost c_fixed: not yet computed (requires phase-payload table +
in-situ cross-check before/after futex). Phase-payload table committed in
phase_payload_table.md.