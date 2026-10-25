#!/usr/bin/env bash
# gpu-sim-jvs: verify that a submitted commit's run.sh works in a clean tree.
# Usage: scripts/verify_committed_run.sh <commit> <bead-id>
set -euo pipefail

commit="${1:?commit required}"
bead="${2:?bead id required}"

repo_root="$(git rev-parse --show-toplevel)"
mkdir -p "${repo_root}/.lab/tmp"
scratch="$(mktemp -d "${repo_root}/.lab/tmp/scratch.XXXXXX")"
trap 'git -C "$repo_root" worktree remove --force "$scratch" 2>/dev/null || true; rm -rf "$scratch"' EXIT

git -C "$repo_root" worktree add --detach "$scratch" "$commit" >/dev/null 2>&1
runsh="$scratch/experiments/$bead/run.sh"
if [[ ! -f "$runsh" ]]; then
    echo "verify_committed_run: FAIL: $runsh not found" >&2
    exit 1
fi
# GPU-hinted skip rule (gpu-sim-jvs): if run.sh is marked GPU-hinted, skip.
if head -n 1 "$runsh" | grep -qi "GPU-HINTED" || grep -qiE "gpu.hinted|gpu-hinted|# *GPU-HINTED|ts --max-runtime|gpqueue" "$runsh"; then
    echo "verify_committed_run: SKIP (GPU-hinted): $bead @ $commit"
    exit 0
fi
if ! (cd "$scratch" && bash "experiments/$bead/run.sh"); then
    echo "verify_committed_run: FAIL: run.sh failed in clean tree at $commit" >&2
    exit 1
fi
echo "verify_committed_run: OK: $bead @ $commit runs in clean tree"
