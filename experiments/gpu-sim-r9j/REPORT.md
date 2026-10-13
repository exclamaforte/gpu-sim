# gpu-sim-r9j REPORT: session-ID collision verdict

## Verdict: NOT exploitable -- verified non-defect, no rename

The three role ids are declared identically by 10 repos, and 15 live `pi
--session-id` processes share them right now. That sharing is harmless:
**every** session-id resolution path in the daemon is repo-scoped, and pi's
own session store is project-scoped by cwd. The evidence is machine-checked
(`evidence.json` via `gather_evidence.py`; `tests/test_session_scope.py`
5/5 PASS with a demonstrated-red negative control).

Per the bead DESIGN step 2, no fix is chosen because none is needed -- and
per the bead's own warning, no session id was renamed (a rename while live
sessions hold the old name is exactly the destructive mistake the bead
cites). Acceptance items 2/4 (cross-repo fix) do not trigger; item 1
(exploitability statement with grep evidence) and item 3 (regression test)
are delivered here.

## The classification (acceptance 1)

| # | site (pi-research-lab/lab.py) | scope | mechanism |
|---|---|---|---|
| 1 | `worker_argv` --session-id (l.972) | repo-scoped | process identity only; pi's store keys sessions by (cwd-slug, id) |
| 2 | bridge connect-back (l.1086, env l.2913) | repo-scoped | unix socket `LAB/<role>.sock`, LAB = repo/.lab |
| 3 | `find_session_file` (l.3836) | repo-scoped | global glob **filtered by header cwd == ROOT** |
| 4-7 | its 4 callers (ll.3979/3999/4039/4091) | repo-scoped | consume the filtered result |
| 8 | TmuxHost sessions (l.677ff) | repo-scoped | `lab-<role>` on private server `LAB/tmux.sock` |
| 9 | HerdrHost tabs (l.787ff) | repo-scoped | persisted tab/pane ids in per-repo labd state |
| 10 | CLI->daemon routing (l.82) | repo-scoped | `LAB = find_root()/.lab`, root from cwd |
| 11 | lab-bridge.ts (246 lines) | repo-scoped | **zero** session-id tokens (socket protocol only) |
| 12 | process/host checks | n/a | `_pid_is_labd` matches daemon entrypoint, `terminate_group` uses process groups -- no name-based session resolution anywhere |

## The two live measurements that settle it

1. **Attach semantics exist but are project-scoped.** Two `pi -p` runs with
   a throwaway id share one transcript (second run remembered QUAGGA) --
   proving `--session-id` attaches globally. BUT the store path is
   `sessions/<cwd-slug>/*_<id>.jsonl`: right now there are **11 session
   files per role id, each with a distinct header cwd** (`one_file_per_cwd:
   true`). Same id + different repo = different sessions. The attach path
   cannot cross repos.
2. **No bare-id lookup exists.** The only transcript lookup
   (`find_session_file`) returns the first glob match **whose cwd is ROOT**;
   the negative control (below) shows the same fixture resolving to 2 files
   without the filter.

## On the observed symptom (acceptance context, not a second claim)

The "[implementer] not connected despite live process" + restart/status/bd
disagreement cluster is consistent with same-repo causes already on the
record (bridge disconnect under load-75 CPU saturation; bd claims are
expiring leases -- this session itself saw `heartbeat: not claimable` then
reclaimed). No cross-repo mechanism is needed, and none exists in the code
paths above. Stated as consistency, not proof of the negative.

## Regression test (acceptance 3)

`tests/test_session_scope.py` (CPU, 5/5 PASS) imports the real lab.py with
fixture-directed ROOT and session store:
- same id under two project slugs resolves only the ROOT-cwd file;
- a foreign ROOT resolves None (never another repo's transcript);
- **negative control**: the bare glob returns both files (ambiguous), and a
  scratch run of lab.py with the cwd-filter line deleted goes red
  (`RED AS DESIGNED: bare resolution ambiguous`) -- so the test guards the
  filter rather than passing vacuously;
- lab-bridge.ts carries no session-id token; worker socket derives from LAB.

If a future change adds a global session-id lookup, test 1 fails (two
candidates, first glob hit need not be ours). If anyone re-declares sharing
*plus* such a path, the fixture catches it -- which is the bead's item 3.

## What was NOT done, and why

- No config renamed in any repo (bead: do not rename unilaterally; 15 live
  processes hold the current names).
- No change to pi-research-lab (separate repo, out of implementer scope;
  and nothing there needs changing).
- pi-research-lab-router/lab.py differs from lab.py but no router daemon is
  running (all `_daemon` processes are pi-research-lab/lab.py) -- noted for
  the central decision, not acted on.
- The daemon's own comment ("a respawn on the same --session-id must never
  overlap the old writer", `terminate_group`) flags same-id writer overlap
  as hazardous *within* a repo under slow shutdown -- worth knowing, outside
  this bead's cross-repo question.
