# gpu-sim-ppcp: the PP term sends the full sequence, ignoring the CP shard

Found by a dimension audit that started from the K/V-width bug (gpu-sim-kvcp): after fixing the CP
width, the obvious next question was "does any other term use the wrong width?".

## The bug

`score()` (and the `comm_bytes` helper) price a pipeline hand-off as

    act_size = seq * d * dtype_bytes

but CP splits the sequence across cp ranks *before* PP stages hand off, so the tensor on the wire is
`(seq/cp) x d`. Every config with **both pp>1 and cp>1** overstates PP traffic by **cp**.

## Why this one is bigger than the K/V fix

The K/V correction moved margins. This one moves the ANSWER at N>=16, because it makes deep PP
combined with CP cheap:

| | published (pre-fix) | with the fix |
|---|---|---|
| N=16 seq 8192 | (1,4,1,4,2) 66.86 ms, margin 0.02% | **(1,8,2,1,0) 62.56 ms, margin 2.16%** |
| N=16 seq 16384 | (2,2,4,1) 137.22 ms | **(1,4,4,1,0) 132.81 ms, margin 0.29%** |
| N=16 seq 32768 | (2,2,4,1) 331.27 ms | **(1,2,8,1,0) 322.46 ms, margin 0.24%** |
| N=32 seq 8192 | (2,4,4,1) 53.31 ms, margin 4.19% | **(2,4,4,1) 41.23 ms, margin 2.25%** |
| N=32 seq 16384 | (1,4,1,8) 104.38 ms | **(1,4,8,1,0) 80.82 ms, margin 4.62%** |
| N=32 seq 32768 | (1,4,1,8) 230.64 ms | **(1,4,8,1,0) 183.07 ms, margin 7.39%** |

N=8 is untouched (every N=8 winner has pp=1 or cp=1). Two qualitative consequences:

1. **The N=16 "tie" mostly dissolves** -- it becomes a 2.16% win for (tp=1 pp=8 cp=2 dp=1), a config
   shape that never appeared in the published answer. The recommendation at 16 GPUs changes, not just
   its confidence.
2. **Long-sequence N=32 drops DP entirely** and prefers pp=4 with cp=8 (183.07 ms at 32768 vs 230.64
   published), a 21% improvement that the old pricing hid.

## Status: FILED, NOT APPLIED

Applying it changes the published winner at four of the nine grid cells plus several headline numbers,
so it needs the same full pass the K/V correction got: regenerate facts, rewrite the affected
paragraphs, refresh pinned tests, regenerate the claims register. `model_fix.diff` and
`facts_with_ppcp.json` are the inputs for that pass.

It should be verified against a harness trace before being trusted at N>=16: the crude
`microbatches = max(pp,1)` message count is a separate modelling assumption that this audit did NOT
validate, and the new winners lean on PP harder than the old ones did.

## Harness check (run after filing) -- the fix should NOT be applied alone

Two findings, both from a real pp=4 / cp=1 trace now frozen in `harness/pp4_cp1.jsonl`:

**1. The harness refuses cp>1 together with pp>1.** tools/torch_parallel/train.py exits with
"--cp>1 with --pp>1 not verified in v1". Every config the width fix promotes at N>=16 -- (1,8,2,1,0),
(1,4,4,1,0), (1,2,8,1,0), (1,4,8,1,0) -- is exactly that combination. So the ppcp fix would put the
published answer at N>=16 onto a code path this harness has never executed, and it cannot be validated
here by construction.

**2. The PP message SIZE is also wrong, and the fix does not address it.** Measured at pp=4/d=256/
seq=512/batch=8: middle ranks send 6 messages per step (3 boundaries x 2 microbatches, over 2 steps),
each of 2,097,152 B. That is `mb(4) x seq x d_model x 4B` -- the harness splits the BATCH into
microbatches, so the wire carries a slice of rows. The model has **no batch dimension at all**; it
prices `seq x d` and assumes `microbatches = max(pp, 1)`. The message COUNT happens to match
(3 boundaries x 4 assumed microbatches = 12 per step per direction for middle ranks... against 6
measured per step, so even the count is off by the microbatch convention), and the payload is off by
the micro-batch size.

**Ruling:** the PP term needs a batch/microbatch-aware redesign validated against the frozen pp=4
trace, not a one-line width fix. Applying `model_fix.diff` alone would promote exactly the configs
whose traffic this harness cannot measure, while leaving the payload error in place -- two wrongs that
partly cancel in the current numbers and would not cancel in the new ones.
