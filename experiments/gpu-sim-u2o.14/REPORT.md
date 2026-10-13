# gpu-sim-u2o.14 REPORT: stage-1 runtime attachment (48-cell dose)

## Outcome

The attachment landed: `dose_measured.jsonl` pairs 48 stage-1 spec ids
(kernel_hash) to measured trial rows (median runtime, ai_cv, queue_state,
protocol). `dose_summary.json`: 48/48, 0 shortfall.

## Dose rule (stated)

8 elementwise ops x n {2^12, 2^18} x block {64, 256, 1024}, full-coverage
grid. Stratified corners, not cherry-picked. kernel_hash via the canonical
`kernel_hash_for` (9yt, on master).

## What happened on the GPU

- Build (CPU, unqueued): 48 specs, nvcc 48/48 exit 0 (`dose_manifest.json`).
- Measure (queued short smoke `20261013t055317-c914a9cb`): all 48 cells
  measured, 2 invocations x 20 trials, GPQUEUE_PROGRESS every 10 rows --
  then the job exited 1 stamping the artifact: `corpus_protocol` did not
  know the `dose_14` id. Diagnosis: my bug, invented protocol, unregistered.
  No measurement lost (raw file held all 48 rows; crash was post-loop).
- Fix (same f6y precedent: register, don't relabel): added a `dose_14`
  registry entry describing the procedure (one binary per kernel via
  synthesize(), 2x20, SafeHarness, SCALE ONLY, may not pool). Stamping these
  `single_run_*` would have claimed the wrong stage -- the registry's own
  reason for separate ids. All 48 rows survived in build/dose_raw.jsonl
  (the crash was at the post-loop stamp, after the raw file closed).
- Merge (CPU): `merge_dose_14.py` asserts one row per manifest hash and
  single-protocol, writes the paired artifact. No requeue needed.

## Reading the numbers honestly

48/48 launch_bound, 0 measurable, median ai_cv 0.025. These are 4K/256K
elementwise cells: launch-dominated by construction (6.6.1/7.1 physics).
The attachment is spec->runtime pairs for the launch regime, which is what
a runtime predictor must handle -- not evidence about the >=10us regime.
ai_cv 0.025 vs 0vk's 0.0018 reflects the same regime difference, not a
measurement defect.

## Gates

- 48/48 rows, hashes join manifest 1:1, single protocol `dose_14`.
- ai_cv on all rows (2 invocations each); own_job_id + queue_state on all.
- `run.sh` replays merge byte-identical.
