# gpu-sim-u2o.13.5 REPORT: selector shipped, retrospective negative, loop gated

CPU-only bead, no GPU.

## Shipped (real, tested)

epic_u2o/acquisition.py: topk_by_variance (total var = head var + sigma
floor), stratified variant, overlap_rate. Unit contracts green.

## Retrospective (real head outputs, NEGATIVE finding)

11.2's 57 per-row predictions (branch-committed, read cross-branch
read-only): variance-top-10 vs error-top-10 overlap 0.2 (~chance 0.175);
failure-triple 7/10 vs 28/57 base (p~0.2, n.s.). With the current
over-conservative head, uncertainty sampling prioritizes no better than
chance. Implication (stated, not buried): fix calibration BEFORE closing
the loop; the loop itself (measure, retrain, re-rank) additionally needs
a GPU campaign. Both halves gated, each with its reason.

## Correction during the run (mine)

First ad-hoc read used RAW var (4/10 overlap, 10/10 triple) and would
have made a stronger claim; the module contract (total var incl. floor)
gives 2/10 + 7/10. Reported the contract numbers, pinned them, and
recorded the discrepancy -- the floor is load-bearing in acquisition,
not just training.

## Backing doc

deferred.md: only the 13.5 placeholder replaced. deferred.md is now
placeholder-free (13.1-13.5 all have sections).
