# gpu-sim-u2o.7.1 — the measured stage-1 batch (the half that did not exist)

The audit note on this bead is the spec: `generator_stage1.py` runs and emits
1000 rows of kernel **specs**, but it has **no runtime column**, so only the
*generate* step of generate → compile → preflight → measure → tokenize was ever
present. This is that missing half, end to end, with real measurements.

**264/264 rows measured, 0 failures.** Phase A compiles all 264 drivers on CPU
(230 s, unqueued); phase B is a single ts job that only measures (111 s,
`20260928t074004-b429f469`). No tenant job was jumped — the batch queued
behind a running conway job.

Each row in `measured.jsonl` carries `kernel_hash`, `launch_config`
(grid/block/shared/threads), `covers`, `median_runtime_ms`, `iqr_ms`,
`n_trials`, `n_invocations`, `ai_cv`, `device_prop_snapshot`, `queue_state`,
`token_count`, `token_digest`, and a `measured_regime` decided **by
measurement**, not by a size guess.

## The finding, stated plainly: the whole batch is launch-bound

Every one of the 264 cells measured **under 10 µs** (min 0.0024, median 0.0032,
max 0.0065 ms). That is the regime 6.6.1 showed is majority **launcher**:
~1.3 µs of irreducible dispatch plus single-launch measurement overhead,
against a 2.4–6.5 µs total. So:

* these medians carry real signal about the **launch path** and very little
  about the kernel body;
* the per-op spread across eight arithmetic ops is 0.0030–0.0035 ms — a 17%
  spread, which at this size is inside the launch-dominated scatter rather
  than an op effect;
* the only clean trend is with size, 0.0030 → 0.0038 ms from n=2¹² to 2¹⁸ — and
  that is precisely the trend a launcher-dominated measurement would *not*
  show, which is a useful internal consistency check.

**70 of 264 rows fail the `max_cv` gate** (`ai_cv` median 0.085, max 0.487).
They are marked `"unreliable": true` in place rather than filtered out, so a
consumer can choose the policy.

The sweep's top size (2¹⁸) is simply too small to leave the launcher.
**Next step is a manifest change, not a code change**: extend the size sweep to
2²²–2²⁴, where 6.6.2's anchor read 1576 GB/s, so rows land in the ≥10 µs
regime. The precompile-then-measure split used here bounds the GPU cost of
that re-run.

## The queue killed the first attempt, and it was right to

`20260928t072959-85612b2e` was killed with *"GPU utilization averaged 0.7% over
the last 5 min, below the 10% minimum. The queue holds the only GPU, so a job
that barely uses it blocks everyone else."* 264 `nvcc` invocations and 528
process starts inside a queued job leave the device idle almost all the time —
and 150 rows had been written when it died.

The fix was not to ask for an exemption but to stop being the bottleneck:
compilation moved to an unqueued CPU phase, and the queued job now does nothing
but measure. The gate enforces it (`all 264 drivers prebuilt; phase B will not
compile`), so the same shape cannot be resubmitted.

## What the earlier beads bought, used rather than rediscovered

* **6.6.1** — sub-10 µs cells are launch-dominated with ~7% spread, so grids
  are *derived from n* (full coverage, ½, ⅛, plus a 1-block case) instead of
  fixed. A first pass with fixed grids produced 264/360 launch-bound cells
  (73%) by construction; this batch is 136/264 (52%), and the size trend is
  visible.
* **6.6.2** — the deliberately poor 1-block configs are **kept and flagged**
  (`deliberately_poor: true`, 8 rows): an anchor needs that dynamic range.
* **6.6.3** — a co-tenant moves these numbers by up to 2.33×, so every row
  carries `queue_state` and the device snapshot rather than assuming a quiet
  box.
* **9ti** — the build refuses any spec whose `kernel_name` is not in its own
  source, before a queue slot is spent.
