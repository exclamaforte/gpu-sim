#!/usr/bin/env bash
# gpu-sim-u2o.20 (widened 2026-10-03): merge-readiness hygiene check.
#
# Assert that no UNTRACKED path in a worktree also exists as a TRACKED path
# in the target branch, and report differing ones by name -- BEFORE any rebase
# is attempted.
#
# The incident this exists for: the 6.6.6.2-r2 worktree carried 47 untracked
# files, 34 of them tracked on master, TWO byte-differing
# (epic_u2o/harness.md, epic_u2o/preflight_validator.py). A stale worktree
# holding untracked copies of files the target now tracks is the same hazard
# as untracked build sources with a sharper edge: an in-place rebase or
# checkout there silently clobbers or refuses on them, and the failure looks
# like a merge conflict rather than a stale-copy problem.
#
# READ-ONLY. This script never stages, stashes, checks out, or deletes
# anything: it reports and exits nonzero. (A check that can destroy the thing
# it inspects is not a check.)
#
# Run: bash tests/assert_worktree_hygiene.sh --target <ref> [--worktree <path>]
#   --target is REQUIRED and has no default: the target branch is exactly the
#   ambient context this lab keeps inheriting wrong (gpu-sim-edu.1.1), so it
#   must be named every time. --worktree defaults to the current directory and
#   is echoed back, so a wrong CWD is visible rather than silent.
set -uo pipefail

TARGET=""
WORKTREE="$PWD"

while [ $# -gt 0 ]; do
    case "$1" in
        --target)   TARGET="${2:?--target needs a ref}"; shift 2 ;;
        --worktree) WORKTREE="${2:?--worktree needs a path}"; shift 2 ;;
        -h|--help)
            echo "usage: $0 --target <ref> [--worktree <path>]"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)"; exit 2 ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "HYGIENE CHECK REFUSED: --target is required (no default target)."
    echo "Name the branch you intend to merge or rebase onto, e.g. --target master."
    exit 2
fi

echo "worktree: $WORKTREE"
echo "target:   $TARGET"

if ! git -C "$WORKTREE" rev-parse --verify --quiet "$TARGET^{commit}" >/dev/null; then
    echo "HYGIENE CHECK REFUSED: target '$TARGET' does not resolve to a commit"
    echo "in this repository. Name it explicitly; ambient refs are not targets."
    exit 2
fi

shadows=0
differing=0
untracked=0

while IFS= read -r line; do
    case "$line" in
        '?? '*) path="${line#?? }" ;;
        *) continue ;;
    esac
    untracked=$((untracked + 1))
    if git -C "$WORKTREE" cat-file -e "$TARGET:$path" 2>/dev/null; then
        shadows=$((shadows + 1))
        if git -C "$WORKTREE" show "$TARGET:$path" 2>/dev/null \
                | cmp -s -- "$WORKTREE/$path" -; then
            echo "  SHADOW (identical) $path -- tracked on $TARGET, untracked here, same bytes"
        else
            differing=$((differing + 1))
            echo "  SHADOW (DIFFERING)  $path -- tracked on $TARGET, untracked here, BYTES DIFFER"
        fi
    fi
done < <(git -C "$WORKTREE" status --porcelain=v1 --untracked-files=all)

echo
echo "untracked paths: $untracked, shadowing $TARGET: $shadows ($differing differing)"
if [ "$shadows" -ne 0 ]; then
    echo "WORKTREE HYGIENE FAILED: $shadows untracked path(s) shadow tracked content on $TARGET."
    echo "Do not rebase or merge here until each is staged, removed, or reconciled --"
    echo "a rebase onto $TARGET will clobber or refuse on them and look like a conflict."
    exit 1
fi
echo "WORKTREE HYGIENE OK: no untracked path shadows $TARGET."
