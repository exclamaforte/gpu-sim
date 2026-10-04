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

## Duplication, stated rather than fixed

`TELEMETRY_FIELDS`/`telemetry()` now exist in THREE near-identical copies:
`measure_noise.py`, `measure_family.py`, and this branch's `verify_harness.py`
(same 9 fields, different order; the branch adds per-field recovery and
`throttled()`). The bead says "same helper ... reuse, do not reinvent", but
no shared module exists to reuse, and unifying two merged experiment scripts
plus this branch is a refactor of validated code outside this bead. Left for
the PI to schedule; unifying is safe now that all three agree on the set.

## Not done, on purpose

6.1.1 is NOT re-run (criterion 5): repeating a validated closed run to fix a
record is the wrong trade on a shared GPU. The GPU proof stands on the
recorded jobs; this bead wires FUTURE runs.
