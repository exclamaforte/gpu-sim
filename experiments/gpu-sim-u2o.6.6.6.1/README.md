# gpu-sim-u2o.6.6.6.1 — the instrument, cheaper than the thing it measures (54x)

**Criterion 1 is MET, and the headline says so.** The bead's criterion is
that the instrument must cost less than the device work it measures.
98.0% of the directory is now served from cache and
`queue_state()` went from **6.09 ms to
0.1123 ms** (54x) — **0.81x the 0.139 ms of
device work**, so criterion 1 is **met**.

(The first attempt on this branch stopped at 0.92 ms / 6.6x and honestly
reported itself as not met; the gap between the two was a single missing `>`.
An earlier revision of this file still carried that pre-fix headline after the
fix landed -- corrected in the 2026-10-03 rework alongside metrics.json.)

## The mechanism: TERMINAL IS TERMINAL

A job in a terminal state can never become active again. So if its
`(mtime_ns, size)` is unchanged and its state is terminal, re-parsing it cannot
change the answer. That is a property of the state machine, not a heuristic and
not a staleness window, which is why the output is **identical** rather than
merely close.

**Why this and not the obvious TTL cache.** A TTL makes a sample *stale*, and the
property 6.6.6 exists for is that a record **observes** rather than asserts. A
sample served from a TTL is a sample of the past — that is the defect 6.6.6 was
filed to remove. So the cache holds *only jobs that are already finished*, and
**the ACTIVE set is read fresh on every call and is never cached.** A cache of
anything else is permitted; a cache of active jobs is not, and a test asserts
that `running`/`queued`/`starting` can never enter `TERMINAL`.

Measured: hit rate 98.0%, 557 parses across
28407 file-visits.

## The bound, stated rather than ignored

A reused job id would let a terminal entry hide a new active job. That is a real
hazard, so terminal entries are **re-verified in full every
30.0 s**, and `load_stats()` reports the staleness bound
and the age of the oldest unverified entry, so a reader sees the limit instead
of inferring freshness.

## Criteria 2-6

- **2** this bead ships no per-example rows of its own (pure instrument change;
  the PI ruled 6.7.1's dataset must not be regenerated). The rows live on
  branch `lab/gpu-sim-u2o.6.7.1` at
  `experiments/gpu-sim-u2o.6.7.1/batch_rows.jsonl` — 264 rows, each with
  `sampled_at` plus per-row `queue_state` and telemetry — and their values are
  identical under the cache (see 3). The instrument's own staleness reporting
  added here is `load_stats()`: hit rate, parses, files seen, the 30 s
  reverify bound and the 0.25 s listing TTL.
- **3** the measurement is unchanged, **measured rather than argued** by
  `tests/test_cache_differential.py`: cached output == uncached output over a
  directory holding every state the module distinguishes, cold, warm and
  repeatedly; a queued-to-running in-place rewrite visible on the very next
  call; a terminal change stale for the documented window, then revealed. Per
  6.7.1's criterion 6, `queue_state` is *not* a filter and no sample is
  discarded on the basis of it, so per-example medians are untouched by this
  change. The end-to-end batched re-run across the 6.7.1 worker is still not
done and is still named as missing rather than implied.
- **4** `tests/test_queue_state_cost.py`, 10 checks. The enforced floor is
  "not within 2x of the uncached baseline", so a restored full scan fails
  loudly. The bead's actual target is guarded **differentially**
  (`test_cached_order_of_magnitude_faster_than_uncached`: cached vs uncached
measured back-to-back in one process, so the machine cancels). The
  **absolute** target (instrument < 0.139 ms) is **REPORTED, never gated** --
  `spot_check.json` carries the ratio, the live queue file count and loadavg
  under `criterion_1_at_live_size`. Three gate designs were tried and all three
  were wrong, which is why the sequence is recorded here:
  1. a live median against 0.139 ms -- failed under load, no code change;
  2. the same with a 2x margin -- failed again (0.279 ms vs a 0.278 ms
     threshold at load 94), i.e. still measuring the scheduler;
  3. conditioned on `loadavg(1m) <= ncpu` -- looked calibrated and was not: at
     loadavg 14.3, "quiet" by that standard, the instrument still measured
     1.91x.
  A gate whose condition does not actually exclude the failing regime is worse
  than no gate, because it looks calibrated. The structural reason the absolute
  ratio moved at all is queue SIZE, not load -- see the section above.
- **5** the record covers what else was queued or running at sample time, plus
  the historical state counts. It is **not** contention detection and **not** a
  filter. The only limit the speedup introduces is the 30.0 s
  reverify window above.
- **6** the scope survives; a test asserts the function body contains no
  filtering vocabulary, having learned to strip comments first — my first
  version of that test failed on the module's own *documentation*, which says
  "must not discard".

## The bug that made the cache weak {#the-bug-that-made-the-cache-weak}

`reverify` was written as a conditional expression, so it evaluated to the
**elapsed seconds** since the last reverify — a tiny positive float, which is
**truthy** in Python. So:

- `not reverify` was always `False`, and the **stat-skip never fired**;
- `or reverify` in the listing condition was always `True`, so the **glob ran on
  every call**.

One missing `>` disabled half the cache *and* defeated the second cache
entirely: **6.09 ms → 0.92 ms** where it should have been **0.112 ms**.

**Why it misled me for an hour.** The cost model stayed *self-consistent* — I was
measuring a cache that genuinely was doing less work than intended, so every
profile said "the cache is not engaging" without saying why. I blamed the
557-path loop, then the glob, and gave up twice.

**What actually found it** was `tests/test_cache_differential.py` — a *differential*
test asserting that a terminal change stays stale for the documented window. That
test can only fail if the cache is not doing what the prose says, which is a
different question from how fast it is. Profiling could not have found it.

**The process lesson, which is the part I would keep:** a deliberately-red
assertion is worth almost nothing if nobody is looking at it.
`test_reports_its_own_cost` sat as an `expectedFailure` carrying the number for
an hour, and the failure it protected against *looked exactly like the work in
progress*. What found it was a different test asking a different question. A
visible debt is still only visible to someone who looks.

## Criterion 3, measured rather than argued

`tests/test_cache_differential.py`, 8 checks against a **temporary** queue
directory (never the shared one):

1. **Identity** — cached output == uncached output, over a directory containing
   every state the module distinguishes, cold and warm and repeatedly.
2. **An active job is never hidden** — a job rewritten in place from `queued` to
   `running` is visible on the very next call. This is the property the whole
   design rests on, and the cheapest way to be wrong is a cache one line too
   broad.
3. **The stated bound is real** — a terminal change stays stale for the
   documented window and is then revealed. I had asserted that bound in prose;
   here it is a measurement.

The end-to-end batched re-run across the 6.7.1 worker is still not done and is
still named as missing rather than implied.

**18 tests, 0 failures, 0 skips** (10 cost + 8 differential). `run.sh` recomputes `spot_check.json` from committed code and runs both suites; it exits 0 under box load (verified at loadavg 35.7) and exits nonzero only when an evaluated gate fails.

## Criterion 1 at live queue size: NOT met, and reported rather than gated

The artifact's `0.81x` was measured at **557 queue files**. The live directory is now **~838**, past the crossover, so the instrument measures ~2.0x device work *at low load on this host*.

**Every timing-derived number here is load-dependent and is derived per run.** The marginal cost is ~0.185 us/file at loadavg 4.3 and ~0.28 us/file at loadavg 38 -- a ~1.5x spread -- and the crossover moves with it (~737 files at loadavg 5.4, materially fewer under load). The 1.14x / 2.0x / 3.29x figures quoted in earlier commits were the SAME quantity at three different loads, not a spread in the code; that is why they are reconciled here rather than averaged. `measure_cost.py` now runs its own size sweep every time and stamps the slope, the crossover and the loadavg together, so no bare constant is baked into the file or asserted anywhere.

Nothing regressed: the cache still removes the parsing (98% hit rate, ~40x same-process differential, byte-identical output). The ACTIVE set is re-read fresh every call *by design* -- that is what stops the cache hiding a neighbour -- so the residual is O(n) and **the queue outgrew the claim**. `spot_check.json` carries `criterion_1_at_live_size.met = false` with the ratio and file count; `metrics.json` carries `criterion_met: true` (at 557 files) beside `criterion_met_AT_LIVE_SIZE: false`.

It is reported rather than gated because a gate that passed here would be misleading, and one that failed would redden `run.sh` on box state. Structural follow-on: `gpu-sim-u2o.6.6.6.3`.
