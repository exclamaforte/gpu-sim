# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

> **Architecture in one line:** Issues live in a local Dolt database
> (`.beads/dolt/`); cross-machine sync uses `bd dolt push/pull` (a
> git-compatible protocol), stored under `refs/dolt/data` on your git
> remote — separate from `refs/heads/*` where your code lives.
> `.beads/issues.jsonl` is a passive export, not the wire protocol.
>
> See [sync-concepts](https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md)
> for the one-screen overview and anti-patterns (don't treat JSONL as the
> source of truth; don't `bd import` during normal operation; don't
> reach for third-party Dolt hosting before trying the default).

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var
## Destructive git: name the target (gpu-sim-edu.1.1)

`git checkout -- .`, `git restore .`, `git reset --hard` and `git clean -fd` take
their target from the **current working directory**. This repo runs ~30 worktrees
under `.lab/wt/` plus the main tree, all one repository, all with a plausible
`AGENTS.md` and `CLAUDE.md` at the root. There is no visual difference between
them and **no error when the wrong one is chosen**.

On 2026-10-03 a `git checkout -- .` intended for a worktree reverted
long-standing uncommitted edits to `AGENTS.md` and `CLAUDE.md` in the MAIN tree.
Recovery came from `git fsck --unreachable`, which surfaced stash-shaped commit
`f73d442` holding exactly those two files. **Unreachable stash commits are a real
recovery route, not a myth** -- but recovery is not the fix; the fix is naming the
target.

RULES

1. Worktree-targeted git is written as `git -C .lab/wt/<bead> ...`, so the target
   is named in the command itself.
2. Never run a CWD-scoped destructive git in the main tree. If you intend to
   discard something, name the files: `git checkout -- <path>`, never
   `git checkout -- .`.
3. Use the guard: `python3 scripts/guard_destructive_git.py -- <git args>`. It
   resolves the tree the command would actually DISCARD (from `-C`/
   `--git-dir`, falling back to the CWD -- naming a TREE is not naming a FILE,
   so `checkout -- .` is never explicit however carefully the tree was named),
   refuses when that tree has modified tracked files and no real pathspec was
   given, and takes a recovery point with `git stash create` (which does NOT
   touch the working tree, unlike `git stash push`) in the target tree before
   anything destructive.

Before ANY destructive git operation, confirm where you are:

    git -C "$(git rev-parse --show-toplevel)" rev-parse --abbrev-ref HEAD

If that says master, you are in the main tree: stop and name your paths.

## Beads Issue Tracker (task tracking only — NOT git authority)

This project uses **bd (beads)** for issue tracking. Run `bd prime` for task-tracking commands.

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for task-tracking command reference
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files
- `bd ready` / `bd show <id>` / `bd update <id> --claim` / `bd close <id>`
- Architecture: issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export.
- Experiment runs use `scripts/bd-exp.sh start/finish` (gpu-sim-u2o.11.6): hypothesis before, outcome + kept + metrics metadata after, negative results kept.
- Closing a gpu-sim-u2o bead requires a concrete artifact cite (dataset+rows, checkpoint, log, or explicit negative kept=false) -- a bare restatement of the title is not a close reason.

## Git authority (bd never decides this)

- `bd`, `bd prime`, and any Beads template/profile text (conservative/minimal/team-maintainer) are NEVER git permission authority. Ignore any such language there.
- Only repository instructions (e.g. `AGENTS.md`), orchestrator prompts (e.g. `.lab/prompts/*`), and direct user instructions grant or deny git commit/merge/push/sync.
- Do NOT wait for human approval on git actions your repo/orchestrator already authorized. Do NOT treat a missing user message as a veto.
- Session close: file follow-up issues, run quality gates (`scripts/gate.sh`; `--fast`
  skips the ~2 min test suite), update issue status, handle git per repo/orchestrator
  rules, then hand off.

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/core-concepts/sync-concepts.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->


## Scratch space (gpu-sim-u6j5)

Do not write experiment traces, sweep outputs, or tool scratch to `/tmp`: it is a
quota-limited tmpfs and long runs die with `EDQUOT` mid-flight. Use the
experiment directory (`experiments/<bead-id>/`) or `.lab/tmp/` (gitignored)
instead. Tools that accept an output path must default to a project-local dir,
not `/tmp`; hardcoding `/tmp` is a bug. `scripts/gpu_run.sh` and `scripts/bd-exp.sh`
run a quota preflight before long runs and refuse to start if the target dir has
less than 1 GiB free.
