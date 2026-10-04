# gpu-sim-u2o.6.6.5 — rebase onto the merged queue_state + the missing CPU test

## Merge-order dependency: resolved by rebase (2026-10-04)

The board-state note held this branch until 6.6.6 merged, or a rebase onto
it: merging first would have run the telemetry sampler beside the OLD
misattributing queue_state. 6.6.6 and its children have since merged, so
`lab/gpu-sim-u2o.6.6.5` (was `bb2d892` on base `a1d0192`) is rebased onto
master tip. The rebase was clean (one commit, one file).

Compatibility re-verified rather than assumed:

- `qs.queue_state(own_description_substr=..., own_command_substr=...)` matches
  the merged signature (all keyword args with defaults), and every key the
  branch reads (`own_job_id`, `n_other_active`,
  `n_other_running_at_sample_time`, `other_active_jobs`) exists in the new
  return — which additionally carries `own_job_entry`, the 6.6.6 fix. The
  hazard the hold existed for (telemetry beside a misattributing record) is
  gone by construction.
- Worktree hygiene vs master: 34 shadows cleared first (3 differing backed up
  to `/tmp/665-shadows-backup`, rest identical-to-master removed), 0 after.

## Criterion 3 was untested: now `tests/test_verify_harness_telemetry.py`

No test referenced `verify_harness` at all. Three CPU-only checks, no GPU
work (nvidia-smi `--query-gpu` is a driver read, no CUDA context):

- the sampler returns the field set 6.6.3 needs (`utilization.gpu`,
  `power.draw`, `clocks.sm`, `temperature.gpu`,
  `clocks_throttle_reasons.active`); if the query fails the snapshot must
  RECORD why (`query_rejected`/`nvidia_smi_error`), not go silent;
- `throttled()` reads the bitmask as causes (zero/absent → False);
- the sampler stays read-only: no `-lgc`, no persistence change, no sudo.

Negative control: removing `"power.draw"` from `TELEMETRY_FIELDS` fails
exactly `test_expected_field_set`; restored, 3/3 green.

## Duplication, stated CORRECTLY this time (review round 2)

The previous version of this section said all three copies carry "same 9
fields, different order" and concluded unifying is safe. Both halves were
false, and the reviewer caught it by counting: measured, not asserted --

- `verify_harness.py` (this branch): 9 fields
- `measure_noise.py` (merged): 9 fields, the SAME SET (pinned by
  `test_telemetry_set_matches_measure_noise`)
- `measure_family.py` (merged): 5 fields -- `clocks.sm`, `clocks.max.sm`,
  `temperature.gpu`, `clocks_throttle_reasons.active`, `persistence_mode`.
  Missing exactly the detector fields: `power.draw`, `power.limit`,
  `utilization.gpu`, `memory.used`.

So unifying is NOT yet safe, and the "safe now that all three agree" conclusion
is withdrawn. The first step is a PI decision this bead cannot make: whether
`measure_family`'s per-example rows should gain the four fields (a schema
change to the 264-row dataset's future, not a refactor) or whether the
requirement is the intersection. Until that is decided, the handoff claim is
narrowed to what exists: the `verify_harness` sampler shares `measure_noise`'s
9-field set and hands 6.6.3 a working detector for verify_harness-path runs;
the per-example rows it would read for dataset-scale detection do NOT carry
power/utilisation, so that half of the handoff is open, not delivered.

The bead headline's "same helper ... reuse, do not reinvent" is therefore
PARTIALLY met: two of the three samplers agree field-for-field (tested), the
third differs, and the merge is scheduled on the PI's word, not assumed.

## Not done, on purpose

6.1.1 is NOT re-run (criterion 5): repeating a validated closed run to fix a
record is the wrong trade on a shared GPU. The GPU proof stands on the
recorded jobs; this bead wires FUTURE runs.
