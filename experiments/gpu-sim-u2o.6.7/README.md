# gpu-sim-u2o.6.7 — real data-generation throughput

The 6.15 audit reopened this bead because its backing artifact was fabricated.
Both halves are now done: the fabrication is characterised and **refused by
code**, and the real number exists.

No GPU, no ts job. Every number comes from a committed artifact.

## The fabricated artifact, in detail

`runs/logs/throughput_demo_run.jsonl`, 15 rows:

* 10 rows carry the **identical** trial set `[5.0, 5.1, 4.9]`, with identical `median_ms`, `cv` and `duration_sec`;
* **5 rows carry no trials at all** yet report `status: completed`, and the companion `summary_demo_run.json` counts all 15 as completed while its `phases` object is empty `{}`;
* only **10 distinct `kernel_hash`** across 15 rows — they repeat, so they are not identifiers;
* it claimed **357,013 ex/hour**, which is **41.5× the real rate**.

Two corrections to my own first reading, because both mattered for the
diagnosis: I initially said the empty-trial rows reported a fabricated
`median_ms: 5.0`. They report `None` — so no median was invented, but a row
with *no measurement* was still counted as completed. And the file is
**untracked** (`was_tracked_in_git: False`) and read by
nothing, so this is a stray local artifact rather than committed evidence — a
distinction worth keeping, because "fabricated evidence in the repo" and
"fabricated evidence that was never committed" call for different responses.

## The real number, derived twice and cross-checked

From 7.1's 264 measured rows (ts `20260928t074004-b429f469`, 0 failures):

| derivation | ex/s | ex/hour |
|---|---|---|
| harness wall clock | 2.3886 | 8,599 |
| per-row `measured_at` span | 2.3784 | 8,562 |

**Disagreement 0.43%** — the two derivations agree. One
derivation is an assertion; two that agree is evidence. `cross_check()` returns
the comparison rather than averaging it, because averaging two disagreeing
numbers produces a third number that is wrong in a new way.

## The answer to the question the bead was opened for

> *"Track examples/hour through the real-GPU measurement stage as an explicit
> metric from day one, since (unlike cclm) this is expected to be the project's
> actual bottleneck, not model training throughput."*

**100.00% of the wall clock is not the kernel
running.** The GPU is asked to do 0.002% of the
work: **10.47 ms of wall per launch against
0.2 µs of kernel**, over
10,560 launches.

So the bottleneck is **per-example process overhead** — spawn, driver init,
per-row setup — not GPU throughput and not training throughput. The fix is
batching or a persistent worker, **not a faster GPU**, and not any work on
training speed. That also makes gpu-sim-u2o.12.2 answerable: the bottleneck is
confirmed and it is not where optimising training would look.

## Why a throughput number now requires evidence

`epic_u2o/throughput_evidence.py` refuses to produce a rate from: no rows, rows
carrying no measurements, **every row carrying an identical trial set**, rows
with no timestamp, or a non-positive wall clock. The rule is one line:

> A throughput number with no underlying rows is not a throughput number.

The identical-trials check is the one that matters, and it needed a fix to be
worth anything: the first version read only 7.1's `invocation_medians_ms` key,
so the fabricated file was rejected as *having no trials at all* — a schema
mismatch — and the check that identifies hand-authored numbers never ran. A
guard that refuses for the wrong reason leaves its important path untested.
Both keys are read now, and a test pins the reason for refusal.

16 tests, 0 failures, CPU only.
