# gpu-sim-k9w — entry-point test (rework round 2, 2026-10-04)

The reviewer's outstanding item, verbatim: the rework fixed
`scripts/safe_git_add.py` and nothing else, so the four tests that pass repo
in explicitly never exercise the entry point -- reverting `current_repo()`
to `SCRIPT_REPO` would leave the suite green while the guard reported "safe"
on the hazard again.

## What was added

`TestEntryPointChecksTheCallersTree` in `tests/test_safe_git_add.py`
(4 tests, subprocess with cwd set, real argv path):

- `test_cli_refuses_from_inside_the_hazardous_tree`: cwd inside the
  throwaway stale repo -> exit 2, both files named, both kinds named.
- `test_cli_passes_where_the_files_are_our_own`: exit 0 when clean.
- `test_cli_honours_repo_from_elsewhere`: `--repo` pointing at the
  hazardous tree from outside refuses (locks the swallowed-flag fix).
- `test_cli_rejects_bogus_flags`: argparse rejects `--totally-bogus`
  instead of running silently.

No production change (`scripts/safe_git_add.py` untouched, as directed).

## Verification

- `python3 -m unittest tests.test_safe_git_add`: **8 tests, OK**.
- Negative control: `current_repo()` temporarily returned `SCRIPT_REPO`
  (the exact regressed behavior). Result: exactly the new cwd test failed;
  the other 7 passed -- proving the old suite could not see the bug.
  Restored from backup, re-ran green.
- Branch rebased onto master `0109762` cleanly before the work; worktree
  hygiene check vs master passes (0 untracked).
