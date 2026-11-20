#!/usr/bin/env bash
# gpu-sim-e2lf: run the append-only walk in a DISPOSABLE worktree.
#
# Why: the landing gate runs in the integration tree, where HEAD == main by
# construction, and scripts/check_append_only.py refuses to run there (yy6o's
# guard) because a walk in the integration tree mutates the very tree being
# landed.  Running the walk in a throwaway worktree of the same commit keeps the
# check real -- it still runs, and still catches a violation -- while leaving the
# integration tree untouched.  --force is load-bearing and only safe here: in the
# throwaway worktree `git rev-parse main` also resolves to the merge commit, so
# without --force the guard would refuse there too.
#
# Usage: run_append_only_stage.sh [<ref>]     (default: HEAD)
#
# Exit status: the walker's own (0 = pass, non-zero = violation/error).  A
# failure to create the worktree is exit 2, never 0: a stage that cannot run is
# not a stage that passed.
set -u

ref="${1:-HEAD}"

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "append-only stage: not inside a git repository" >&2
  exit 2
}

tmp_base="${GATE_TMP:-$repo_root/.lab/tmp}"
mkdir -p "$tmp_base" 2>/dev/null || {
  echo "append-only stage: cannot create $tmp_base" >&2
  exit 2
}

tmpwt="$tmp_base/appendonly.$$"

cleanup() {
  git -C "$repo_root" worktree remove --force "$tmpwt" >/dev/null 2>&1 || true
  rm -rf "$tmpwt"
  git -C "$repo_root" worktree prune >/dev/null 2>&1 || true
  # Remove the record now that the worktree is gone.
  python3 - "$repo_root" "$tmpwt" <<'PYUNRECORD'
import json, os, sys
root, path = sys.argv[1], sys.argv[2]
rf = os.path.join(root, ".lab", "tmp", "appendonly-worktrees.json")
if not os.path.exists(rf):
    raise SystemExit(0)
try:
    data = json.load(open(rf))
except Exception:
    data = []
data = [r for r in data if r.get("path") != path]
with open(rf, "w") as f:
    json.dump(data, f)
PYUNRECORD
}
# Cleanup on normal exit AND on interruption: a killed gate must not leave a
# registered worktree behind (that is the residue class this bead is about).
trap cleanup EXIT INT TERM HUP

# Record the disposable worktree path so a reaper can remove it by name if
# this process is killed mid-walk (gpu-sim-k3gb).
python3 - "$repo_root" "$tmpwt" "$$" <<'PYRECORD'
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

if ! git -C "$repo_root" worktree add --detach "$tmpwt" "$ref" >/dev/null 2>&1; then
  echo "append-only stage: git worktree add --detach $tmpwt $ref failed" >&2
  exit 2
fi

# Build libfakegpu.so in the disposable worktree (not committed).
( cd "$tmpwt" && make -j2 >/dev/null 2>&1 ) || true

rc=0
( cd "$tmpwt" && python3 scripts/check_append_only.py --force ) || rc=$?
exit "$rc"
