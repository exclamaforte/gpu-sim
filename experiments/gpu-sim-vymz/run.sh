#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 - "$ROOT" << 'PY'
import json, subprocess, os, sys
root = sys.argv[1]
os.chdir(root)
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    return r.stdout, r.stderr, r.returncode

m = json.load(open('experiments/gpu-sim-vymz/metrics.json'))
pruned = set(m['pruned'])
kept = {k['branch']: k['reason'] for k in m['kept']}

# current lab/* branches
out,_,_ = run(['git','branch','--list','lab/*','--format=%(refname:short)'])
current = set(b.strip() for b in out.split() if b.strip())
current.discard('lab/gpu-sim-vymz')

errors = []
# 1. pruned branches must be gone
for b in pruned:
    if b in current:
        errors.append(f"pruned branch still present: {b}")
# 2. kept branches must be present
for b in kept:
    if b not in current:
        errors.append(f"kept branch missing: {b}")
# 3. every remaining branch must be non-closed or have cherry + or dirty worktree
# map worktrees
wt_out,_,_ = run(['git','worktree','list','--porcelain'])
worktrees = {}
cur = None
for line in wt_out.splitlines():
    if line.startswith('worktree '):
        cur = line.split(' ',1)[1]
    elif line.startswith('branch '):
        b = line.split(' ',1)[1].replace('refs/heads/','')
        if cur: worktrees[b]=cur; cur=None

for b in current:
    bead = b.replace('lab/','')
    # bd status
    bo,_,_ = run(['bd','show',bead,'--json'])
    status = None
    try:
        j = json.loads(bo)
        if isinstance(j, list): j = j[0] if j else {}
        status = j.get('status')
    except Exception:
        status = 'unknown'
    # cherry
    co,_,_ = run(['git','cherry','master',b])
    plus = sum(1 for l in co.splitlines() if l.startswith('+'))
    # dirty tracked?
    wt = worktrees.get(b)
    dirty = False
    if wt and os.path.isdir(wt):
        so,_,_ = run(['git','-C',wt,'status','--porcelain'])
        dirty = any(not l.startswith('??') for l in so.splitlines())
    if status == 'closed' and plus == 0 and not dirty:
        errors.append(f"fully-superseded closed clean branch remains: {b}")
# 4. kept reasons fresh check (spot: status+cherry+dirty)
for b, reason in kept.items():
    bead = b.replace('lab/','')
    bo,_,_ = run(['bd','show',bead,'--json'])
    status = None
    try:
        j = json.loads(bo)
        if isinstance(j, list): j = j[0] if j else {}
        status = j.get('status')
    except Exception:
        status = 'unknown'
    co,_,_ = run(['git','cherry','master',b])
    plus = sum(1 for l in co.splitlines() if l.startswith('+'))
    wt = worktrees.get(b)
    dirty = False
    if wt and os.path.isdir(wt):
        so,_,_ = run(['git','-C',wt,'status','--porcelain'])
        dirty = any(not l.startswith('??') for l in so.splitlines())
    if status != 'closed':
        exp = f"not closed ({status})"
    elif plus > 0:
        exp = f"unlanded patches ({plus} +)"
    elif dirty:
        exp = "worktree has uncommitted tracked changes"
    else:
        exp = "other"
    if exp != reason:
        errors.append(f"reason mismatch {b}: recorded '{reason}' vs fresh '{exp}'")

print(f"before (recorded pruned): {len(pruned)}; current remaining lab/* (excl self): {len(current)}")
if errors:
    for e in errors:
        print("FAIL:", e)
    sys.exit(1)
print("OK: prune invariants hold")
PY
