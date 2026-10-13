# gpu-sim-u2o.6.6.6.2 — a row must be able to say how *it* was read

Criterion 2 of `6.6.6.1` asks that every row record **which path produced its
sample** and **the age of the underlying read**, so no reader is told a sample is
fresher than it is.

Before this change that information existed only in `load_stats()`, which is
**process-global**. Two samples taken a minute apart get the same answer, and a
caller who never called `load_stats()` has nothing at all. That is exactly the
gap: the information existed but was attached to the wrong thing.

## Four read paths, and they are not interchangeable

| path | what it means |
|---|---|
| `fresh_parse` | read from disk on this call; age 0 |
| `terminal_hit_stat_ok` | stat'd, signature unchanged, terminal |
| `terminal_hit_no_stat` | served from the terminal cache **without a stat**, so the read can be up to `TERMINAL_REVERIFY_SEC` old |
| `listing_ttl_hit` | the file **list** came from the 0.25 s listing cache, so a job submitted inside that window is not in this sample |

The last two are deliberately **not merged**. A listing-TTL read is stale in a
different way from a terminal-cache read: folding them together would let a
reader believe a sample was fresh when what actually happened is that the file
*list* was a quarter-second old.

The **active set is never cached**, so an active job is always `fresh_parse` with
age 0 — the property `6.6.6.1` exists to protect, now visible per sample rather
than inferred from a global counter.

`read_age_ms` is the age of the **oldest** underlying read among records served —
the worst case for that sample, not a flattering average.

## Additive, and proven additive

`sample_provenance` is added to `queue_state()`'s return. Nothing else changed.

`test_byte_identical_to_the_pre_change_code` loads `queue_state.py` **from
master**, runs both versions over the *same* temporary queue, and asserts every
pre-existing key is byte-identical. Comparing the record to itself would prove
nothing about the change; this compares it to the actual previous version.

## It can say nothing happened

An empty queue reports `source: "none"` and zero counts, **not** a confident
`read_age_ms` of `0.0`. A zero would read as "freshly read" when the honest
answer is "read nothing".

## A vestigial line, removed

The first version of the additive test contained `saved = qs.sample_provenance_enabled = True`
— setting an attribute that does not exist and is never read. It passed, because
it did nothing. Removed rather than left as a comment that reads like a control.

## Artifacts

CPU only, no GPU, no `ts` job. All suites use temporary queue directories.

```
experiments/gpu-sim-u2o.6.6.6.2/{run.sh,output.log,job.json}
tests/test_sample_provenance.py                   9 tests
epic_u2o/queue_state.py                           _load_all_with_provenance()
```
