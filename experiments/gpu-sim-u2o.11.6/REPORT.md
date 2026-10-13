# gpu-sim-u2o.11.6 REPORT: protocol exists AND is used

CPU-only bead. The audit's point was use, not existence -- so the
deliverable is existence + one real production use + the close rule.

## Script

scripts/bd-exp.sh: start / finish / board, adapted from cclm's
battle-hardened version (duplicate-active guard, verdict-overwrite guard,
start-signature check, needs-review block) with this lab's metric flags
(--logmae --spearman --kendall --coverage --ece --ex-sec --topk --set)
and a REQUIRED --artifact on finish (the strengthened close rule,
enforced by the tool, not just written down).

## First production use

gpu-sim-u2o.10.4.1: transformer ranking on family_holdout from 11.2's
per-row mus (predictions read cross-branch, read-only). Result neutral,
kept=true: rho 0.8858 / tau 0.7373 / top5 0.4 -- identical to the
feat17 single-feature ranker, so model ranking adds nothing (one more
sequence-earns-nothing data point, recorded on the bead with metadata).
board lists it. This is the first experiment bead anywhere here with
real metrics attached -- the exact absence the audit named.

## Close rule

AGENTS.md now requires every gpu-sim-u2o close to cite a concrete
artifact (dataset+rows, checkpoint, log, explicit negative kept=false).

## Backing doc

experiment.md: only the 11.6 placeholder replaced.
