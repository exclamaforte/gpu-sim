# gpu-sim-9yt REPORT: one kernel_hash definition

## Outcome

One definition, explicit, with the bead's acceptance proof: both corpora
regenerate hash-identical through the unified scripts (7.1: 264/264, e5q:
192/192). `tests/test_kernel_hash_single_definition.py` 3/3 PASS pins it:
AST scan over 154 repo files finds 43 kernel_hash-named sites, all
canonical; the divergent recipe demonstrably mismatches (0/264).

## What was wrong (on master at claim time)

The old `KernelSpec.kernel_hash` (sha256(source)) was already gone, but
FOUR inline recipes still minted `kernel_hash`-named values:

| site | recipe | status |
|---|---|---|
| `experiments/gpu-sim-u2o.7.1/build_batch.py:110` | source+json([grid,block,size]) | same as canonical -- routed through `kernel_hash_for` |
| `experiments/gpu-sim-u2o.6.7.1/batch_measure_worker.py:107` | same (duplicated text to preserve the 7.1 join) | routed through `kernel_hash_for` -- join now by construction |
| `epic_u2o/batch_measure.py:64` | source+json([grid,block]) -- **missing problem_size, a genuinely third definition** | routed through `kernel_hash_for` (no importers, no committed manifests) |
| `epic_u2o/safe_harness.py:778` (measure_spec) | source+json([grid,block,shmem]) -- **divergent, stamped into RunResult.metadata['kernel_hash']** | routed through `kernel_hash_for` (spec carries problem_size; only log names change) |

Plus `safe_harness.py:892` (debug CLI): ephemeral `khash` renamed to
`run_tag` so the name cannot be mistaken for the corpus key. `corpus_miner.py`
(the bead's third site) no longer contains any hash -- already gone.

The bead said "do not simply delete one definition / do not add a third":
nothing was deleted (all call sites now share the one function) and the two
divergent recipes were removed, not added to.

## Acceptance proof (the bead's own words)

> Regenerate both corpora and assert the row hashes still match.

`experiments/gpu-sim-9yt/regen_proof.py` imports each build_batch as a
module, redirects .cu output to scratch (monkeypatched HERE -- the tree is
untouched), runs the real sweeps, and compares against the committed
measured rows (`proof.json`):

- 7.1: regen 264, measured 264, match=true
- e5q: regen 192, measured 192, match=true

Because the 7.1 inline recipe was textually identical to `kernel_hash_for`,
the routing is a no-op numerically -- and the proof shows it rather than
asserting it. The 6.7.1 row-for-row join property is preserved for the same
reason (same bytes in, same hash out).

## Negative control (xy8 practice)

`test_divergent_recipe_mismatches`: the pre-fix divergent recipe (no
problem_size) applied to 7.1's generator output overlaps the corpus 0/264.
A re-keying would be caught -- the regen check is not decoration. (A green
run on fixed code is not the demonstration; the 0/264 on the broken recipe
is.)

## What was NOT done

- No corpus row touched (proof compares against committed measured.jsonl;
  zero content changes outside the listed recipes + test + 9yt artifacts).
- No `source_hash` alias added: nothing needs pure-source dedup today, and
  the bead forbids adding names speculatively.
- `structural_split.py:313` and `6.6.2` digests are content digests under
  different names -- out of scope, left alone.
