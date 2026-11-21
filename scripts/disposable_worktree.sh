#!/usr/bin/env bash
# 4k2x: run a command in a disposable git worktree, and always remove it.
#
# Where the tree lives is not the caller's choice: it is created under
# <repo>/.lab/tmp and removed on every exit path (success, failure, signal).
# A tree in $HOME is how /home/gabe accumulated 19 gpu-sim worktrees
# (clean43, cleanfu2..5, clean7q2..8, cleanenj2/3, cleanmain, cleanqf/2 --
# 13 GB) that no janitor could see and no process owned.
#
# Usage:
#   disposable_worktree.sh [--repo DIR] [--commit REF] -- <command...>
#
# The tree's path is printed on stdout before the command runs, so a caller
# that needs it (e.g. to write a log beside it) can read the first line.
# Exit status is the command's.
set -euo pipefail

repo=""
commit="HEAD"
while [ $# -gt 0 ]; do
  case "$1" in
    --repo)   repo="${2:?--repo needs a directory}"; shift 2 ;;
    --commit) commit="${2:?--commit needs a ref}";    shift 2 ;;
    --)       shift; break ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *)        break ;;
  esac
done
[ $# -gt 0 ] || { echo "disposable_worktree: no command given (use -- <cmd>)" >&2; exit 2; }
[ -n "$repo" ] || repo="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "disposable_worktree: not in a git repo and --repo not given" >&2; exit 2; }
repo="$(cd "$repo" && pwd -P)"

mkdir -p "$repo/.lab/tmp"
scratch_root="$(cd "$repo/.lab/tmp" && pwd -P)"
scratch="$scratch_root/disposable.$$.$RANDOM"
# The invariant, asserted rather than assumed: a scratch tree is always inside
# the repo's own .lab/tmp. If a future edit adds a path argument, this refuses
# anywhere else (including $HOME) instead of leaking again.
case "$scratch" in
  "$scratch_root"/*) ;;
  *) echo "disposable_worktree: refusing scratch outside $scratch_root" >&2; exit 2 ;;
esac

recorder="$repo/scripts/appendonly_worktrees.py"
cleanup() {
  rc="$1"
  git -C "$repo" worktree remove --force "$scratch" >/dev/null 2>&1 || rm -rf "$scratch"
  git -C "$repo" worktree prune >/dev/null 2>&1 || true
  if [ -f "$recorder" ]; then
    python3 "$recorder" unrecord "$scratch" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT TERM HUP

if [ -f "$recorder" ]; then
  python3 "$recorder" record "$scratch" --pid "$$" >/dev/null 2>&1 || true
fi
git -C "$repo" worktree add --detach "$scratch" "$commit" >/dev/null
# shellcheck disable=SC2317  # the trap above always runs
echo "$scratch"
( cd "$scratch" && "$@" )
