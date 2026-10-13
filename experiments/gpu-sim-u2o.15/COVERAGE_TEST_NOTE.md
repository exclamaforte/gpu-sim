# gpu-sim-u2o.15 — the coverage test now calls the code (2026-10-04)

The standing item: `test_an_unproven_query_cannot_certify` built a record by
hand and re-applied the gate's rule inline, so deleting the gate from
`close_requires_artifact.py` left the suite green.

## What changed

- `scripts/close_requires_artifact.py`: extracted `apply_coverage_gate(rec,
  coverage)` from `main()` (identical logic: unproven coverage ->
  `satisfies_the_acceptance = None` + `blocked_reason`). `main()` calls it;
  no behavior change.
- `tests/test_close_requires_artifact.py`: the test blinds the primary query
  (same blinding the neighboring tests use), then runs the REAL
  `gather_with_proven_coverage` + `audit` + `apply_coverage_gate`. With
  nothing seen, `audit` alone would certify (`satisfies True` — asserted as
  the intermediate state, which is the dangerous direction made visible),
  and only the gate stops it.

## Verification

- 19 tests, OK.
- Negative control: gate body replaced with `pass` -> exactly the rewritten
  test fails, other 18 green. Restored from backup, re-ran green.
- Branch rebased onto master tip cleanly; worktree hygiene vs master clean.
- `main()` itself not re-run here: it rewrites `close_audit.json` from the
  live board, and regenerating the audit record is not this item.
