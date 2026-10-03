# gpu-sim-u2o.6.6.2.1.1 — an artifact whose branch moved should say where it went

An artifact stamps the branch it was measured on. That stamp is *true* — the ts
jobs really did run there — but history moves, and a reader who follows it gets a
tree that no longer holds the numbers beside it.

Three artifacts carry a branch stamp. All three were wrong in different ways:

| Bead | Stamped ref | Ref alive? | Content landed on master |
|---|---|---|---|
| `gpu-sim-u2o.6.6.2` (metrics + job) | `lab/gpu-sim-u2o.6.6.2-r2` | **yes, and dirty** | `566ba0b` |
| `gpu-sim-u2o.17.5` (job) | `lab/gpu-sim-u2o.17.5-r2` | **no — deleted** | `ccb32ad` |

17.5 is the worse case and was not in the bead's design: its ref was deleted
outright, so a reader following the stamp finds **nothing at all**. 6.6.2's is
arguably worse in practice: the ref still resolves, still looks authoritative,
and holds a tree that disagrees with the artifact.

## Omission, not null

A writer that cannot name a superseding commit **omits** `superseded_by`.
Writing `"superseded_by": null` would assert that nothing supersedes it — a
claim nobody checked — and a null reads as coverage while an absent key does not
(`gpu-sim-u2o.6.6.6` criterion 5). `stamp_tree_state` pops the key rather than
writing `None`, and a test asserts no artifact in the tree carries one.

## Keyed on where the content landed, not on whether the ref survives

My first implementation short-circuited when the stamped branch still existed,
reasoning that asserting unverified supersession is the same error in the other
direction. **That was wrong for the case that matters**: `6.6.2-r2` still exists
while its content is on master. A live ref *is* the hazard. The rule now keys on
the landing commit and records the ref's continued existence as a fact beside it:

```json
"superseded_by": {
  "branch_ref": "lab/gpu-sim-u2o.6.6.2-r2",
  "branch_still_exists": true,
  "commit": "566ba0b",
  "reachable_from_master": true,
  "note": "the content landed on master at 566ba0b; the stamped branch ref still
           exists, so it is a live-but-not-current tree and a reader following
           the stamp alone would get a tree that disagrees with this artifact"
}
```

## Criterion 2, verified by regeneration rather than by assertion

`merge_metrics.py` now stamps on write. Verified by running it: regenerating
`metrics.json` produced `tree_state.superseded_by.commit = 566ba0b`, and a
flattened key-by-key diff of before/after showed **six keys added, none removed,
none changed** — no measured value moved.

## The check can fail

`test_the_suite_can_actually_find_a_bad_artifact` feeds the same predicate a
deliberately broken artifact. Without it, the criterion-3 assertion could pass
because the detector finds nothing at all — an empty detection set and a clean
tree look identical.

## Criterion 4 is not done here

The dirty `lab/gpu-sim-u2o.6.6.2-r2` worktree is **not** disposed of: disposal
is destructive and is tracked under `gpu-sim-u2o.6.6.2.1` criterion 3. **No
measurement is at risk** — the three v8 family files that mattered were copied
into `experiments/gpu-sim-u2o.6.6.2/variance_probe_v8/` before `6.6.2` merged.

## Artifacts

CPU only, no GPU, no `ts` job.

```
scripts/tree_state.py                        resolve + stamp (never writes null)
tests/test_tree_state_supersession.py        10 tests, whole-tree invariant
experiments/gpu-sim-u2o.6.6.2.1.1/{run.sh,metrics.json,output.log,job.json}
```
