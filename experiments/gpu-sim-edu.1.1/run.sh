#!/usr/bin/env bash
# gpu-sim-edu.1.1 -- CPU only, run directly. Never through ts.
# Proves the guard refuses CWD-scoped destructive git and that the recovery
# route (git stash create -> fsck --unreachable) is real.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
ART="experiments/gpu-sim-edu.1.1"
echo "=== guard demo: $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 - "$ART/output.log" <<'PY'
import subprocess, sys, tempfile, shutil
from pathlib import Path
log = open(sys.argv[1], "w")
d = Path(tempfile.mkdtemp()) / "repo"; d.mkdir()
def g(*a, **k): return subprocess.run(["git","-C",str(d),*a],capture_output=True,text=True,check=k.get("check",True))
g("init","-q"); g("config","user.email","t@e.com"); g("config","user.name","t")
(d/"AGENTS.md").write_text("original\n"); g("add","-A"); g("commit","-qm","base")
(d/"AGENTS.md").write_text("LOCAL EDIT THAT MUST SURVIVE\n")
print("main tree state:", g("status","--porcelain").stdout.strip(), file=log)
for args in (["checkout","--","."], ["restore","."], ["reset","--hard"],
             ["checkout","--","AGENTS.md"], ["-C",str(d),"checkout","--","."]):
    r = subprocess.run(["python3","scripts/guard_destructive_git.py","--repo",str(d),
                        "--check-only","--",*args],capture_output=True,text=True)
    print(f"  {str(args):<44} exit={r.returncode}  {r.stderr.strip().splitlines()[0][:52]}", file=log)
h = subprocess.run(["git","-C",str(d),"stash","create"],capture_output=True,text=True).stdout.strip()
print(f"\nrecovery point: {h}", file=log)
print(f"  working tree still holds the edit: {'MUST SURVIVE' in (d/'AGENTS.md').read_text()}", file=log)
fs = subprocess.run(["git","-C",str(d),"fsck","--unreachable","--no-progress"],capture_output=True,text=True).stdout
print(f"  fsck surfaces it: {h in fs}", file=log)
if h:
    body = subprocess.run(["git","-C",str(d),"show",f"{h}:AGENTS.md"],capture_output=True,text=True).stdout
    print(f"  recovery commit carries the edit: {'MUST SURVIVE' in body}", file=log)
log.close()
print(Path(sys.argv[1]).read_text())
shutil.rmtree(d.parent, ignore_errors=True)
PY
echo "=== test suite ==="
python3 tests/test_destructive_git_guard.py 2>&1 | tee -a "$ART/output.log" | tail -4
rc=${PIPESTATUS[0]}; echo "=== tests exit=$rc ==="; [ "$rc" -eq 0 ] || exit "$rc"
echo "=== run.sh: all green ==="
