# gpu-sim-u2o.6.6.6.3 — the residual sweep, measured and PARTLY fixed

**Acceptance (sub-linear growth) is NOT met. This is a partial result.**

Every number below is **quoted from `sweep.json`**, which `measure_sweep.py`
regenerates at run time against temporary directories and stamps with the load
it was taken at. This file carries no figures of its own; `run.sh` rewrites the
quoted lines below from the artifact, and `tests/test_finding_quotes_sweep.py`
fails if a number here is not in `sweep.json`.

<!-- BEGIN QUOTED FROM sweep.json -->
loadavg(1m)=41.15 at measurement time; every figure below is load-dependent and not comparable with a sweep taken at another load.

| files | median ms | us/file | x device budget |
|---|---|---|---|
| 200 | 0.0847 | 0.4234 | 0.61x |
| 400 | 0.1862 | 0.4656 | 1.34x |
| 700 | 0.2797 | 0.3996 | 2.01x |
| 862 | 0.3392 | 0.3936 | 2.44x |
| 1200 | 0.4764 | 0.3970 | 3.43x |
| 2000 | 0.7995 | 0.3998 | 5.75x |
| 4000 | 1.9086 | 0.4772 | 13.73x |

Slope 0.45402 us/file. Per-file spread 1.212x -> linear=True.

Attempt-2 gain vs master (interleaved A/B, same fixture and load, master@16e0c9c):
| files | branch ms | master ms | gain |
|---|---|---|---|
| 200 | 0.0962 | 0.1045 | 1.09x |
| 400 | 0.1694 | 0.1849 | 1.09x |
| 700 | 0.2953 | 0.3240 | 1.10x |
| 862 | 0.3488 | 0.3868 | 1.11x |
| 1200 | 0.4859 | 0.5906 | 1.22x |
| 2000 | 0.8967 | 1.2920 | 1.44x |
| 4000 | 2.7489 | 4.3649 | 1.59x |

Median gain 1.109x (gain > 1 means this branch is faster).
<!-- END QUOTED FROM sweep.json -->

## Why the numbers are quoted and not written

The hand-typed-number-beside-a-derived-one failure is what this bead's sibling
6.6.6.1 spent four review rounds on, and what 6.11.1's lint exists to catch. A
number written here is a number that will drift the next time the box is
loaded differently. The artifact is the single home; this file quotes it.

## The finding

The cost is **linear** in directory size and is **not I/O**. The existing
`reverify` branch already skips `stat()` for cached terminal entries, so the
residual is per-file Python work — a dict lookup and an append, once per file.
That distinction is what decides the shape of any fix.

## Attempt 1: memoise the whole sweep — rejected

The fastest change available, and it is wrong. `tests/test_cache_differential.py`
caught it: `test_a_running_job_is_never_served_stale` and
`test_queued_to_running_is_visible_immediately` both fail.

The reason is structural and is the whole constraint: the **ACTIVE set is
precisely what must never be cached** — caching it is the neighbour-hiding
defect 6.6.6 exists to remove — so it is re-read on every call by design. A job
going `queued` → `running` changes with no file added and no generation bump, so
a memoised sweep cannot see it. That is not a defect in the memoisation; it is
the property the 6.6.6.1 review round explicitly validated.

A 130× speedup that silently reintroduced a four-bead-old defect is the worst
outcome available here, and the differential suite is the only reason it did not
ship. **Not claimed.**

## Attempt 2 (committed): terminal in bulk, active always fresh — modest

Terminal records are served from a bulk map with no `stat()` and no cache
lookup; every file whose last known state was ACTIVE, and every file never
seen, is re-read on this call. The guarantee is preserved exactly — verified
`queued` → `running` in place is seen on the very next call.

The gain is real but small, because the dominant cost is the per-file loop
itself and the bulk map only removes work that was already being skipped. The
measured per-size gains are quoted from `sweep.json` in the table block above,
from the interleaved branch-vs-master arm that shares each fixture and each load.

Two silent bugs were found and fixed on the way, both worth recording:
  * the terminal subset was first built by indexing `out` against the file list,
    but files that fail to `stat()` or parse are skipped, so `out[i]` and
    `_FILE_LIST[1][i]` are not in correspondence and a zip pairs the wrong
    record with the wrong path;
  * `terminal_snapshot` was only populated on the bulk path, which itself
    requires a prior snapshot, so it stayed empty forever and the optimisation
    did nothing. The tests passed throughout while the feature was inert —
    the green-and-useless failure, and the reason this bead measures rather
    than trusting the suite.
  * the rework dropped the unparseable-file guard that master carries
    (6.6.6.3.1): a permanently malformed file costs three parse retries
    with sleeps on every call. The live queue holds exactly one such file,
    so every call paid that sleep budget and three cost tests failed as
    "the terminal cache is probably not engaging". Restored the guard
    from master unchanged: a seen-bad file gets one parse and no sleeps
    per call, a rewrite heals the entry, and the skip is counted in
    provenance rather than silent.

## Not done

The criterion is sub-linear growth per call; this branch does not achieve it. A
set-membership partition with a preassembled terminal list is the named next step
and is not done.
