# gpu-sim-u2o.6.6.2.1 — a finished job's artifact was overwritten, silently

On 2026-09-28 job `20260928t053729-5a28768b` finished, exit 0, and its log ended
`wrote .../family_launch.json`, `18/18 configs measured`. Twenty seconds later
something shaped like a git restore replaced that file with the committed
version. On disk sat a **coherent but older** measurement with nothing in it to
say it had been replaced. The 18/18 launch numbers survived only in a log.

That is the mixed-provenance shape this project was reopened over, arriving from
a different direction: not a fabricated number, but a real number deleted and
replaced by an older real one.

## What now prevents a silent repeat

`measure_family.py` stamps every artifact it writes with an
`artifact_provenance` block: the **job id** that wrote it, the time, and a
**sha256 of the content** (computed over the artifact *excluding* that block, so
it can be recomputed afterwards).

`verify_artifact_provenance.py` then checks three things per artifact:

1. **provenance exists** — an older artifact with no block is reported
   `NO_PROVENANCE`, not treated as a pass. Absence of evidence is not evidence
   of integrity.
2. **content matches its own hash** — a restore changes the content, so the
   hash stops matching. This catches the incident even though the restored file
   is perfectly well-formed.
3. **the job id matches the queue's own record** — the queue is the authority
   on what ran; the artifact is only a claim.

It **reports and never modifies**. A checker that repairs the thing it checks
cannot be used to check it.

## The regression test states the trap

`test_the_restored_file_still_parses_and_looks_fine` asserts explicitly that
the restored artifact **parses, has the expected keys, and carries plausible
numbers** — and is still caught. Without that assertion a reader might believe
the check works because it parses things. It does not parse; it **compares**.

## A correction to the record, including to me

The bead's design says **two** of the 34 master-owned copies in the stale
worktree differ (`harness.md`, `preflight_validator.py`). **Measured, it is
twenty**, all under `epic_u2o/`. Those two files are genuinely in the set; they
are not the only ones.

I reported "TWO differing" myself in an earlier comment on `gpu-sim-u2o.6.6.2`.
I had confirmed those two files differed and reported the number of examples I
had checked as the total. Twenty stale copies of master-owned code is a
different problem from two, and the number that goes in a disposition is the
one that changes what you do about it.

## Items 2 and 4: reported, not decided

- **Item 2** (the 5 modified tracked files) is left **unresolved on purpose**.
  Committing them means regenerating `metrics.json` from a *mixed* set — v8
  memory/reduce/gemm beside v6 launch — which is the very artifact this bead
  exists to prevent. Resetting them discards three finished GPU jobs. Both need
  your word; the triage script records the choice as deliberately unmade.
- **Item 4** (quiescence before git operations that touch tracked paths) is
  enforced for *destructive* git by `scripts/guard_destructive_git.py`, which
  also refuses anything that would delete the lab runtime. It is **not** yet
  enforced for non-destructive tracked-path writes, and nothing here pretends
  otherwise.

## Artifacts

CPU only, no GPU, no `ts` job.

```
experiments/gpu-sim-u2o.6.6.2/verify_artifact_provenance.py   the checker
experiments/gpu-sim-u2o.6.6.2/triage_stale_worktree.py       items 2 and 3
experiments/gpu-sim-u2o.6.6.2/{provenance_check,worktree_triage}.json
tests/test_artifact_provenance.py                           5 tests, temp files only
```
