# gpu-sim-smqu: archive then purge duplicate closed audit-violation beads

Verdict: **DONE**. 715 duplicate closed audit beads purged after a full export; the
tracker went from 1135 rows to 420, and the non-audit bead set is byte-identical
before and after.

## Why this bead exists

`audit violation: session` was minted every ~5 min by the liveness audit — 665 closed
beads all pointing at one audit page, 59% of the tracker, each one paid for on every
`bd list`/`bd show`/`bd ready` scan. The dedupe fix (a standing bead that is reopened
instead of a new one minted) landed 2026-10-31 and works; these are the pre-fix
duplicates.

## What was done

1. Export first, always: `audit_violations_purged.json` holds one record per purged
   bead (id, title, status, created_at, closed_at, close_reason, description) plus a
   header naming the kept beads and the counts. 715 records.
2. `bd delete --from-file <ids> --force`, dry-run first (`Would delete: 664 issues`,
   0 dependencies, 0 labels), then two passes: 664 `session` + 51 `daemon`.
3. Verify by set difference, not by count:
   - non-audit ids: 416 before, 416 after, `diff` empty (`ids_before.txt` /
     `ids_after.txt`);
   - deletion arithmetic: 664 + 51 = 715 = 1135 - 420;
   - 0 dependency links, 0 labels, 0 text references touched (so nothing else
     referred to them);
   - no title remains with more than 5 occurrences.

## Kept

| bead | why |
|---|---|
| `gpu-sim-cjmn` | the standing session-violation bead the audit lattice now reuses (occurrence+1) |
| `gpu-sim-s3n1` | newest daemon-violation bead, kept as the standing record |
| `gpu-sim-3uln`, `gpu-sim-qklt` | the two `role` violations; below any duplicate threshold |

## Note on execution

This bead was first closed by another role with close_reason `Closed` and **no
artifact, with all 665 duplicates still present**. It was reopened and executed by the
PI as tracker maintenance. The rule it violated is the repo's own: a close needs a
concrete artifact cite.

Artifacts: `metrics.json`, `audit_violations_purged.json`, `ids_before.txt`,
`ids_after.txt`.
