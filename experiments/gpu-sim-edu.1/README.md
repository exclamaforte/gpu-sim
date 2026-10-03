# gpu-sim-edu.1 — a dispatch list that cannot offer finished work

Twice in one session the top of `bd ready` was work an implementer could not
do: `gpu-sim-u2o.6.11.1` was already MERGE READY, and `gpu-sim-u2o.6.6.1` was
frozen awaiting a verdict. Both cost a session to unpick, and the second nearly
produced a rework cycle spent redoing items that had already been cleared.

**One of the two suspected classes turned out to be real.** That is the
headline, and it is worth more than having a fix for both.

| Class | Status | Evidence |
|---|---|---|
| validated-but-unmerged | **REAL, fixed here** | `bd ready` listed 6 beads carrying `validated` / `validated-pending-merge` |
| dep-blocked-but-open | **NOT REAL — bd is already correct** | once deps are recorded as `type=blocks`, `bd ready` excludes them; no code change needed |

So this tool does **not** reimplement `bd`'s dependency logic. It *verifies* it,
so a regression is caught rather than assumed, and adds the label filter `bd`
genuinely lacks. Reimplementing blocker detection would be a second source of
truth for something `bd` gets right, and would be wrong the first time `bd`
changed its semantics.

## My own probe was wrong first, and that is recorded

The first dep-blocked probe reported **8 leaks**. It was my bug: dependency
sub-records carry no `status`, so every edge looked open, and `bd list` omits
closed issues by default. Resolved properly, every blocker was `closed`. Had I
reported the first number I would have filed a fix for a defect that does not
exist — and this lab has spent a night learning that a structural argument
needs an empirical check.

## The rule

Exclude a bead carrying a merge-pending label **and** having no outstanding
verdict. The second clause is not decoration: a bead under review carries
`needs-review`, and it is *not* finished even if it also says `validated`.
Merging those classes would hide work in progress — verified by
`test_outstanding_verdict_beats_a_merge_pending_label`.

Ordering is priority first, then `created_at`, then id. Explicitly **not**
recency: recency is what put a reviewed-and-merge-ready bead at the top of a
dispatch in the first place. `bd` already sorts by priority; this makes the
tiebreak deterministic rather than accidental.

## The filter must be able to pass

From `gpu-sim-u2o.6.11.1`: *a check that only ever reports disagreement is
indistinguishable from one whose accounting was simply wrong.* A filter that
dropped every bead would satisfy every exclusion assertion in the suite and be
useless in production. So `tests/test_bd_ready_filter.py` asserts ordinary
ready work **survives** — plain beads, `needs-rework`-only beads, beads whose
only blocker is *closed*, and parent-child edges — plus
`test_the_filter_does_not_empty_the_list`, which fails if the offered count
collapses. An empty dispatch and a correct dispatch are indistinguishable
downstream, so that assertion is the guard.

## Criterion 4: this bead is itself a victim of the inheritance defect

It was created with `--parent gpu-sim-edu` and would have inherited
`needs-rework`. `bd create` now offers `--no-inherit-labels`; the inherited
label was removed right after creation and
`measure_dispatch.py` checks the bead's own labels as a computed property
rather than an assertion.

## What is NOT in this repo

`bd config` has no ready-filter namespace and `bd ready` exposes
`--exclude-label` but no persistent default, so the filter cannot be made the
default from here. **Pointing the lab daemon at `scripts/bd_ready_filtered.py`
is the one step outside this repo, and it is the PI's to make.** Until then this
is a working tool with a passing suite, not a change in behaviour.

## Artifacts

`run.sh` recomputes `metrics.json` from the live board and runs both suites,
failing nonzero on any failure. CPU only, no GPU, no `ts` job.

```
experiments/gpu-sim-edu.1/measure_dispatch.py   computes every number; none typed
experiments/gpu-sim-edu.1/run.sh               recompute + suite, nonzero on failure
experiments/gpu-sim-edu.1/{metrics,output.log,job.json}
scripts/bd_ready_filtered.py                   the filter
tests/test_bd_ready_filter.py                  12 tests, fixture + live database
```