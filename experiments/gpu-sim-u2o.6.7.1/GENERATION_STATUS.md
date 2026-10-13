# Generation status: BUILT, MEASUREMENT BLOCKED BY QUEUE CITIZENSHIP

**No corpus was produced.** The acceptance is written, the protocol guard is
implemented and tested, the generator builds on CPU in 12 s, and the
measurement has been submitted twice and killed twice by the shared queue for low
GPU utilisation. That is a finding about the SPEC, not a harness bug, and it is
written here rather than left for the next person to rediscover.

## What is done

* **gpu-sim-u2o.7.1 acceptance items 12–15 written FIRST**, as ruled, with the
  figures (label range, family count, protocol, the frozen reference), and
  verified: the field went 5026 → 9601 chars with all of the PI's items 1–11
  intact alongside them. The field write went through
  `scripts/bd_field_guard.py`; a first attempt was refused and the field was
  confirmed intact before anything was retried.
* **`epic_u2o/corpus_protocol.py`** — every row carries a protocol id;
  `write_rows()` refuses an unlabelled row; `assert_single_protocol()` refuses a
  mixed corpus; `poolable()` returns nothing unless the caller names a target,
  so pooling is a decision at the call site. 20 tests.
* **`corpus_decision.json`** — the corpus sized against what 9.x must LEARN: the
  binding constraint is the label dynamic range (≥ 3 orders of magnitude; the
  264 rows span 2.70×, a log10 spread of 0.431, which is fatal for 9.3's
  noise-aware loss on log-runtime) and the family count (≥ 24; 7.7's family
  holdout is not a real test with 8).
* **`generate_scale_corpus.py`** — 24 structural families × 9 sizes (2^12–2^28) ×
  2 block sizes × 3–4 launch configs = **1656 cells**, all 24 families
  synthesizing. Builds on CPU in **12 s**, with build and run split so the
  shared GPU never sees a compiler.

## What blocks it, precisely

| job | outcome | GPU utilisation |
|---|---|---|
| `20260928t145612-006810af` | `killed_underused` | **9.8%** over 5 min |
| `20260928t155843-af8aedc6` | `killed_underused` | **6.2%** over 5 min |

Both were the measurement-only job, prebuilt, with no compiler in the path. The
killed combined build+run job was the first: 24 `nvcc` invocations dominated a
job whose device work is about half a second.

**The finding: the spec produces a corpus the shared queue will not accept.** The
regime we most need coverage of — launch-bound kernels at 2^12–2^18, measured in
2–10 µs — is by construction the regime that leaves the GPU idle. Per spec there
are ~50 launches of a few microseconds, so device work per example is ~100 µs
while the host does a dispatch, a `queue_state()` read and an NVML sample. The
apparatus, not the kernel, is the load.

This is the same shape as the 6.6.6.1 result one level up: **the instrument must
be cheaper than the thing it instruments.** Per-example sampling costs more than
the kernel at this end of the size range, and that is a property of the spec,
not of the code.

## What has to be decided before another submission

1. **Raise device work per example without touching the trial count.** The
   protocol ruling fixes 2 invocations × 20 timed launches, so the lever is the
   *number of examples per job* and how they are batched, not the trials.
2. **Or accept a low-utilisation job and run it off the shared GPU** — which
   needs a human decision, since the queue exists to stop exactly this.
3. **Or split the corpus**: a small launch-bound set measured under the existing
   arrangement, and a memory-bound set (2^24–2^28) whose device work per
   example is large enough to keep the queue happy. The second set is also the
   one `6.6.4` says the roofline can only be validated on, so it may be the
   more valuable half anyway.

## Two defects of mine in this generator, recorded because both looked like success

* **Two writers, one path.** The reader thread and the row loop both wrote
  `corpus_scale.jsonl`; on budget exhaustion the row loop never ran, so the
  "corpus" was 12,429 raw `BEGIN`/`TRIAL`/`END` lines wearing a corpus
  filename. The stream is an input file now, and only the row loop writes the
  corpus.
* **All-or-nothing write.** The first version collected the whole stdout and
  wrote rows at the end, so the 15-minute timeout lost all 1656 rows and
  reported nothing. Rows are now stamped and flushed per example, so a budget
  stop keeps every spec that completed.
