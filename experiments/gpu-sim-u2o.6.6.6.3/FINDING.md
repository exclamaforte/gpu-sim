# gpu-sim-u2o.6.6.6.3 — the residual sweep, measured and PARTLY fixed

**Acceptance (sub-linear growth) is NOT met. This is a partial result.**

Every number below is **quoted from `sweep.json`**, which `measure_sweep.py`
regenerates at run time against temporary directories and stamps with the load
it was taken at. This file carries no figures of its own; `run.sh` rewrites the
quoted lines below from the artifact, and `tests/test_finding_quotes_sweep.py`
fails if a number here is not in `sweep.json`.

<!-- BEGIN QUOTED FROM sweep.json -->
loadavg(1m)=6.45 at measurement time; every figure below is load-dependent and not comparable with a sweep taken at another load.

| files | median ms | us/file | x device budget |
|---|---|---|---|
| 200 | 0.0540 | 0.2699 | 0.39x |
| 400 | 0.1031 | 0.2576 | 0.74x |
| 700 | 0.1820 | 0.2600 | 1.31x |
| 862 | 0.2191 | 0.2541 | 1.58x |
| 1200 | 0.3022 | 0.2518 | 2.17x |
| 2000 | 0.5057 | 0.2529 | 3.64x |
| 4000 | 1.0402 | 0.2601 | 7.48x |

Slope 0.25808 us/file. Per-file spread 1.072x -> linear=True.
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
exact figure is measured by `run.sh`, not asserted here.

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

## Not done

The criterion is sub-linear growth per call; this branch does not achieve it. A
set-membership partition with a preassembled terminal list is the named next step
and is not done.
