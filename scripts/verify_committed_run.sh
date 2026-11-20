#!/usr/bin/env bash
# gpu-sim-jvs: verify that a submitted commit's run.sh works in a clean tree.
# Usage: scripts/verify_committed_run.sh <commit> <bead-id>
set -euo pipefail

commit="${1:?commit required}"
bead="${2:?bead id required}"

repo_root="$(git rev-parse --show-toplevel)"
common_dir="$(git rev-parse --git-common-dir)"
main_root="$(cd "$common_dir/.." && pwd)"
mkdir -p "${repo_root}/.lab/tmp"
mkdir -p "${main_root}/.lab/tmp"
scratch="$(mktemp -d "${repo_root}/.lab/tmp/scratch.XXXXXX")"
trap 'git -C "$repo_root" worktree remove --force "$scratch" 2>/dev/null || true; rm -rf "$scratch"; python3 - "$main_root" "$scratch" <<PYUNRECORD 2>/dev/null || true
import json, os, sys
root, path = sys.argv[1], sys.argv[2]
rf = os.path.join(root, ".lab", "tmp", "appendonly-worktrees.json")
if os.path.exists(rf):
    try:
        data = json.load(open(rf))
    except Exception:
        data = []
    data = [r for r in data if r.get("path") != path]
    with open(rf, "w") as f:
        json.dump(data, f)
PYUNRECORD' EXIT

# record the disposable worktree path (gpu-sim-k3gb)
python3 - "$main_root" "$scratch" "$$" <<'PYRECORD'
import json, os, sys
root, path, pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
rf = os.path.join(root, ".lab", "tmp", "appendonly-worktrees.json")
os.makedirs(os.path.dirname(rf), exist_ok=True)
try:
    data = json.load(open(rf)) if os.path.exists(rf) else []
except Exception:
    data = []
data.append({"path": path, "pid": pid})
with open(rf, "w") as f:
    json.dump(data, f)
PYRECORD
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
