# gpu-sim-u2o.6.6.6.3 -- the residual sweep, measured and PARTLY fixed

**Acceptance (sub-linear growth) is NOT met. This is a partial result and the
artifact says so.** Branch `lab/gpu-sim-u2o.6.6.6.3`.

## Measured: the cost is linear and it is NOT I/O

Warm, cached terminal entries, median per call:

    files   200    400    700    862   1200   2000   4000
    ms    .0603  .1081  .1993  .2510  .3365  .5877  1.2143
    us/file .301   .270   .285   .291   .280   .294   .304

Flat per-file: ~0.28 us/file. The live queue is 862 files, so the instrument
sits at ~1.8-2.0x the 0.139 ms device budget. The crossover the bead predicted
(~739 files) is behind us and the queue keeps growing.

Crucially the cost is **not** stat() calls: the existing `not reverify` branch
already skips stat() for cached terminal entries. It is pure per-file Python
work -- a dict lookup and an append, 862 times.

## Attempt 1: memoise the whole sweep. 130x faster, and WRONG.

    files   200    400    700    862   1200   2000
    ms    .0012  .0014  .0017  .0019  .0022  .0029     (0.01x device budget)

`tests/test_cache_differential.py` caught it immediately:
`test_a_running_job_is_never_served_stale` and
`test_queued_to_running_is_visible_immediately` both failed.

The reason is structural and is the whole constraint: **the ACTIVE set is
precisely what must never be cached** -- caching it is the neighbour-hiding
defect 6.6.6 exists to remove -- so it is re-read every call by design. A job
that goes `queued` -> `running` changes with no file added and no generation
bump, so a memoised sweep cannot see it. That is not a bug in my memoisation;
it is the property 6.6.6.1's review round explicitly validated.

I nearly shipped a 130x speedup that would have silently reintroduced the exact
defect four beads ago. The differential suite is the only reason it did not ship.

## Attempt 2: terminal in bulk, active always fresh. Correct, modest.

Terminal records are served from a bulk map with no stat and no cache lookup;
every file whose last known state was ACTIVE, and every file never seen, is
re-read on this call. The guarantee is preserved exactly:

    queued -> running, in place: seen on the very next call

Cost at 862 files: 0.2775 -> 0.2242 ms. Real, but only ~1.24x, because the
dominant cost is the 862-iteration LOOP itself, and the bulk map only removes
work that was already being skipped.

Two bugs found and fixed on the way, both silent:
  * the terminal subset was first built by indexing `out` against the file list,
    but files that fail to stat or parse are skipped, so `out[i]` and
    `_FILE_LIST[1][i]` are not in correspondence and a zip pairs the wrong
    record with the wrong path;
  * `terminal_snapshot` was only populated on the bulk path, which requires a
    prior snapshot, so it stayed empty forever and the optimisation did nothing.

## What is left, named

Partition by SET membership instead of per-file dict lookup, and concatenate a
preassembled terminal record list:

    active = [p for p in _FILE_LIST[1] if p not in terminal_set]   # O(active)
    out    = terminal_records + [read(p) for p in active]          # memcpy + O(active)

That removes the per-file dict-get-and-append for the ~848 terminal files and
should land near 0.03 us/file, roughly 8-10x rather than 1.24x. It keeps the
active set re-read on every call, so the guarantee is unchanged.

NOT DONE, and not claimed. The bead's acceptance criterion is sub-linear growth
per call; this branch does not achieve it.
