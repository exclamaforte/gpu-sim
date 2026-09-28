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

| signal | uncontended | queued-routed | unqueued | discriminates |
|---|---|---|---|---|
| power.draw | 76.9 W | 574.3 W | 573.7 W | yes |
| utilization.gpu | 6% | 97% | 98% | yes |
| temperature.gpu | 40 °C | 59 °C | 60 °C | yes |
| clocks.sm | 2490 MHz | 2745 MHz | 2760 MHz | yes |
| memory.used | 425 MiB | 1101 MiB | 1101 MiB | yes |
| clocks_throttle_reasons.active | 0x0 | 0x4 | 0x4 | yes **here** |

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
  `temperature.gpu > 50 °C` — uncontended reads 26–77 W / 6% / 38–40 °C,
  contended 574 W / 97–98% / 57–60 °C, so an order-of-magnitude gap makes the
  threshold insensitive rather than delicate; **or**
* `clocks.sm < 0.75 ×` the max SM clock seen in the same run (idle floor read
  195–300 MHz against 2490–2760 MHz under load).

Keep the queue-state record as the primary channel for queue-routed
contention, telemetry as the primary channel for the rest.

## Residual risk, stated rather than smoothed over

Light contention that moves none of power, utilisation, temperature or clocks
is undetectable by **any** channel available here — neither telemetry nor
`queue_state`. For that case the only defences are the estimator itself (the
three-invocation median plus the `max_cv` gate, which the CV column above
shows is doing real work) and process discipline. I would rather hand 6.6 and
9.3 an uncomfortable boundary than a false sense that the record is complete.

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
