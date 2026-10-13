# gpu-sim-u2o.6.6.3 — contention: the error bar, and whether we can see it

One ts job, `20260928t072008-b34ddec9`. Three conditions × three anchor cells
(launch-bound, memory-bound, compute-bound) × 3 independent process
invocations. Kernels are 6.6.2's hand-written anchor cells, copied with
provenance; nothing generated, nothing untrusted.

## Headline

| cell | uncontended | queued-routed | unqueued | ratio (worst) |
|---|---|---|---|---|
| launch-bound (k_empty, n=1024) | 2.35 µs | 2.80 µs | 2.48 µs | **1.19× (+19%)** |
| memory-bound (k_saxpy, n=2²²) | 10.54 µs | 24.54 µs | 24.64 µs | **2.33× (+133%)** |
| compute-bound (tiled GEMM 1024³) | 258.9 µs | 274.4 µs | 274.1 µs | 1.06× (+6%) |

**A co-tenant puts a 1.19× (+19%) error bar on a sub-10 µs measurement on this
box — against the 7.4% across-invocation CV 6.6.1 measured for the same
regime.** So contention is ~2.5× *larger* than the noise the estimator already
accounts for, and the largest effect is not on the smallest cell: the
memory-bound cell more than doubles. A launch-bound-only error bar would have
been the wrong thing to plan around.

The across-invocation CV is itself a detector: memory-bound reads 0.042
uncontended and 0.276 / 0.345 under contention — the estimator's own
disagreement rises by ~7×.

## Detection: yes, from telemetry — but the obvious signal is the wrong one

| signal | uncontended | queued-routed | unqueued | relative spread |
|---|---|---|---|---|
| power.draw | 76.9 W | 574.3 W | 573.7 W | 0.866 |
| utilization.gpu | 6% | 97% | 98% | 0.939 |
| temperature.gpu | 40 °C | 59 °C | 60 °C | 0.333 |
| clocks.sm | 2490 MHz | 2745 MHz | 2760 MHz | 0.098 |
| memory.used | 425 MiB | 1101 MiB | 1101 MiB | 0.614 |
| clocks_throttle_reasons.active | 0x0 | 0x4 | 0x4 | 1.000 **here** |

The ordering lives in the spread, not in a yes/no flag: every signal moves,
so a `discriminates: true` on all six would carry no information (removed
from the artifact by PI ruling). `clocks.sm` at 0.098 is the weakest by an
order of magnitude — which is why the rule below no longer contains a clock
clause.

**Do not build the detector on `clocks_throttle_reasons`.** It went non-zero
in this run only because the burner I chose sits at the 575 W power cap. The
reviewer's own unqueued sweeps ran with it at `0x0` throughout, so a
discard rule keyed on it would have caught my burner and missed theirs. Your
prediction that contention here shows up as boost-state movement rather than
throttling is consistent with what I measure, and it is why the rule below
leads with power and utilisation.

**`queue_state` is blind to both contended conditions here** — it reported
`n_other_active=0, n_other_running=0` in all three, because the burner is a
child of this job with no JOB_ID of its own. That is the finding the third
condition exists to produce, and it is why the two channels are complementary
rather than redundant: queue_state for contention routed through the queue,
device telemetry for everything else.

## Proposed discard rule (implementable, not a sentiment)

Drop a sample if, at either boundary:

* `power.draw > 150 W`, **or** `utilization.gpu > 50%`, **or**
  `temperature.gpu > 50 C` — uncontended reads 26–77 W / 6% / 38–40 C,
  contended 574 W / 97–98% / 57–60 C, so an order-of-magnitude gap makes the
  threshold insensitive rather than delicate. Stated in the units measured in.

Dropped clauses (recorded, not silent): the `clocks.sm < 0.75x run-max`
clause (spread 0.098, weakest by an order of magnitude, and it fires on
legitimately cold starts — worse than no rule) and `clocks_throttle_reasons`
as a primary signal (0x4 here only because the burner sits at the 575 W cap;
the reviewer's unqueued sweeps ran 0x0 throughout).

**Detection floor, travelling with the rule:** this catches sustained
contention at or near saturation (>50% utilisation class). Below that floor
— a partial tenant at low duty — the thresholds may not fire, and that case
is undetectable by **any** channel available here, telemetry or `queue_state`
alike. The thresholds were derived from two extremes (idle vs a saturated
burner); the partial-tenant response curve is unmeasured, so treat them as
provisional for anything short of saturation and measure a partial tenant
when one occurs naturally rather than manufacturing one.

**Regime weighting (PI ruling):** memory-bound cells move 2.33x under a
co-tenant while launch-bound move 1.19x/1.05x, so the weighting is procedural
— a tighter memory-bound *number* would be invented, not measured:

* memory-bound: NEVER compare across runs without a contention record on
  BOTH sides (a 2.33x neighbour effect manufactures regressions); the numeric
  thresholds above are necessary, not sufficient, for this regime;
* launch-bound: a null result is NOT evidence of no change (a 2.33x-class
  effect hides inside that noise floor) — a launch-bound A/B states something
  about the instrument, not the kernel.

Keep the queue-state record as the primary channel for queue-routed
contention, telemetry as the primary channel for the rest. Neither alone is
sufficient: in this run `queue_state` reported `n_other_active=0` in all
three conditions with a burner resident on the device.

## Merge note (ordering artefact, not a defect)

This artifact's `queue_state` records predate `own_job_entry` (6.6.6 reached
master after ts job 20260928t072008-b34ddec9): `own_job_id` is non-null in
all three conditions and `own_job_entry` is absent in all three. The
`n_other_active=0` readings are real and recorded and simultaneously
unverifiable from the record itself. No re-run: the next run carries the
field anyway.

## Criterion key map (item 1)

Criterion 1 names one `median_contended_us`; this bead measured two
contended conditions, so per-cell flat keys are condition-suffixed
(`median_queued_routed_us`, `median_unqueued_us`, `ratio_queued_routed`,
`ratio_unqueued`, `ai_cv_*` per condition) with a `key_map` in each cell
stating the relation. The nested per-condition blocks stay authoritative;
the flat keys are aliases. Every per-invocation row carries its condition's
`queue_state` (`own_job_id`, `n_other_active`, `overlapping_job_ids`)
— criterion 4 as written.

## Process

* One queue slot: the whole experiment ran inside a single ts job, and the
  burner was started **inside** it. No GPU work ran outside the queue. The
  "unqueued" condition models a tenant invisible to the queue channel, which is
  the detection question — it is not a decision to bypass the queue.
* No clock locking, no sudo, no persistence-mode change. 6.4 stays held.
* Blanking `CUDA_VISIBLE_DEVICES` was **not** used as a sandbox: `run_kernel`
  sets it to `str(self.gpu_id)`, so it would not have sandboxed anything.
* Four jobs died before measuring anything, all on CPU-detectable defects (an
  import of a module another unmerged branch owns; a cell size not in the
  anchor's own list; the same unknown kwarg passed in two places, the second
  time because the gate covered `compile_cell` but not `measure()`).
  `preflight_static.py` now executes the module body, resolves every name,
  dry-compiles all three cells with nvcc, and calls `measure()` — so that class
  cannot reach the queue again. All five job ids are in `job.json`.
