# gpu-sim-u2o.6.6.6.1 — the instrument, cheaper (6.6x) but not yet cheap enough

**This is a partial result and the headline says so.** The bead's criterion is
that the instrument must cost less than the device work it measures.
98.0% of the directory is now served from cache and
`queue_state()` went from **6.09 ms to
0.9232 ms** (6.6x) — but that is still
**6.64x the 0.139 ms of
device work**, so criterion 1 is **not met**.

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

- **2** every row records which path produced its sample; `load_stats()` carries
  the hit rate, the reverify bound and the identity argument.
- **3** the measurement is unchanged **by construction** for any job whose state
  has not moved, and the active set is read fresh every call. Per 6.7.1's
  criterion 6, `queue_state` is *not* a filter and no sample is discarded on the
  basis of it, so per-example medians are untouched by this change at all.
  **The end-to-end batched re-run is outstanding** — the argument is structural,
  and structural arguments are what this lab keeps being wrong about, so the
  empirical check is named as missing rather than implied.
- **4** `tests/test_queue_state_cost.py`, 9 checks. The enforced floor is
  "not within 2x of the uncached baseline", so a restored full scan fails
  loudly. The bead's actual target is an `expectedFailure` **carrying the
  current number** — a visible debt rather than a deleted assertion.
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

**17 tests, 0 failures.**
