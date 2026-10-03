# gpu-sim-u2o.6.3 — the disk verdict is older than the disk work

Branch `lab/gpu-sim-u2o.6.3-rb`, built by cherry-picking 6.3's two content
commits onto current `master` (`45c3b27`).

## Why not a rebase

The PI's merge prohibition was correct and is now structurally impossible to
repeat: merging the old branch would have reverted 6.1.1's file-derived
`sys.path` fix, because that branch's `kernel_driver.py` carried the hardcoded
absolute path. Two of its commits were themselves rebases dated **2026-09-28**,
so its history could not be rebased again in place.

Rebasing it in place also fails for an unrelated reason worth recording: the
old worktree holds 47 untracked files, all of which are now TRACKED on master, so
a branch switch would overwrite them. Cherry-picking onto master sidesteps both.

## The verdict predates the disk work

| | |
|---|---|
| reviewer verdict (disk half "not delivered") | 2026-09-27 |
| `1b70682` "per-run scratch dirs, no leak on failure, a real mid-run disk bound" | 2026-09-28 |

Verified present on this branch:

- **per-run identity** — `ScratchDir.__init__` builds
  `{counter:04d}-{pid}-{time_ns}`, so no two runs share a directory;
- **no leak on the failure path** — cleanup is `__exit__`, which fires on
  exception, not only on success;
- **mid-run bound** — `start_monitor()` samples the directory *while the child
  runs*; `preflight()` only checks before and after, which is what the verdict
  said was missing;
- **per-file backstop** — `RLIMIT_FSIZE` via `preexec()`.

`tests/test_disk_budget.py` passes.

## The 6.1.1 revert is gone

`git diff master..HEAD -- epic_u2o/kernel_driver.py` is **12 insertions, 0
deletions**: purely additive. The file-derived `sys.path.insert` sits at
kernel_driver.py:164, and the `processes_limit` rationale survives in
`resource_limits.py` fact 2, including the measurement the PI said was at risk
("NPROC=32 (and 512) made the CUDA driver fail its first thread creation").

## Known pre-existing failure

`tests/test_preflight_cannot_pass.py` fails one check, "libfakegpu is
resolvable for the smoke path". Verified identical on bare `master` with no 6.3
work applied: it is the `gpu-sim-u2o.6.3`-independent `gpu-sim-u2o.20` defect
(libfakegpu.so is a build artifact absent from clean worktrees), not a
regression.
