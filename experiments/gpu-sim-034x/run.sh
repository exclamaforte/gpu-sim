#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
MB=$(git merge-base HEAD master)
SCRATCH="$ROOT/.lab/tmp/034x/scratch"
METRICS="$ROOT/experiments/gpu-sim-034x/metrics.json"
OUTLOG="$ROOT/experiments/gpu-sim-034x/output.log"

mkdir -p "$ROOT/.lab/tmp/034x"
rm -rf "$SCRATCH"
git clone --local "$ROOT" "$SCRATCH" >/dev/null 2>&1
cd "$SCRATCH"

# We'll collect output into OUTLOG
: > "$OUTLOG"

{
  echo "=== Run 1: clean (no unretired losses) ==="
} >> "$OUTLOG"

# Run 1: clean
cd "$SCRATCH"
git checkout -- . >/dev/null 2>&1
set +e
out1=$(python3 scripts/check_append_only.py --base $MB 2>&1); rc1=$?
set -e
echo "$out1" >> "$OUTLOG"
echo "rc=$rc1" >> "$OUTLOG"
echo "" >> "$OUTLOG"
if [ "$rc1" -ne 0 ]; then echo "FAIL: run 1 expected rc=0 got $rc1"; exit 1; fi

# Run 2: unretired deletion (fabric_band_robustness removed)
{
  echo "=== Run 2: unretired deletion (fabric_band_robustness removed) ==="
} >> "$OUTLOG"
python3 - <<'PY'
import json
p="docs/model_facts.json"
d=json.load(open(p))
del d["fabric_band_robustness"]
json.dump(d, open(p,"w"), indent=2)
open(p,"a").write("\n")
PY
set +e
out2=$(python3 scripts/check_append_only.py --base $MB 2>&1); rc2=$?
set -e
echo "$out2" >> "$OUTLOG"
echo "rc=$rc2" >> "$OUTLOG"
echo "" >> "$OUTLOG"
if [ "$rc2" -ne 1 ]; then echo "FAIL: run 2 expected rc=1 got $rc2"; exit 1; fi
if ! echo "$out2" | grep -q "fabric_band_robustness"; then echo "FAIL: run 2 did not name fabric_band_robustness"; exit 1; fi

# Run 3: retired deletion (fabric_band_robustness retired with reason+commit)
{
  echo "=== Run 3: retired deletion (fabric_band_robustness retired with reason+commit) ==="
} >> "$OUTLOG"
# restore from git first
git checkout -- docs/model_facts.json >/dev/null 2>&1
python3 - <<'PY'
import json, subprocess
p="docs/model_facts.json"
d=json.load(open(p))
del d["fabric_band_robustness"]
commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
d.setdefault("_retired_fact_blocks",[])
d["_retired_fact_blocks"].append({"key":"fabric_band_robustness","reason":"test retirement","commit":commit})
json.dump(d, open(p,"w"), indent=2)
open(p,"a").write("\n")
PY
set +e
out3=$(python3 scripts/check_append_only.py --base $MB 2>&1); rc3=$?
set -e
echo "$out3" >> "$OUTLOG"
echo "rc=$rc3" >> "$OUTLOG"
if [ "$rc3" -ne 0 ]; then echo "FAIL: run 3 expected rc=0 got $rc3"; exit 1; fi

# Restore scratch working tree
git checkout -- . >/dev/null 2>&1

# Recompute counts and deep-compare to metrics.json
cd "$ROOT"
python3 - <<'PY'
import json, subprocess
ROOT="."
def show(ref, path):
    r=subprocess.run(["git","-C",ROOT,"show",f"{ref}:{path}"],capture_output=True,text=True)
    return r.stdout if r.returncode==0 else None
def count_mf(text):
    d=json.loads(text); return len([k for k in d if k!="_retired_fact_blocks"])
def count_ledger(text):
    import glob, os
    ddir="tests/fs2y_ledger.d"
    if os.path.isdir(ddir):
        return len(glob.glob(os.path.join(ddir,"*.json")))
    if not text:
        return 0
    d=json.loads(text); return len(d.get("entries",[]))
mb=subprocess.check_output(["git","-C",ROOT,"merge-base","HEAD","master"],text=True).strip()
base_mf=count_mf(show(mb,"docs/model_facts.json"))
head_mf=count_mf(open("docs/model_facts.json").read())
base_ld=count_ledger(show(mb,"tests/fs2y_ledger.json"))
head_ld=count_ledger("")

# added/removed keys
base_mf_keys=set(json.loads(show(mb,"docs/model_facts.json"))) - {"_retired_fact_blocks"}
head_mf_keys=set(json.loads(open("docs/model_facts.json").read())) - {"_retired_fact_blocks"}
base_ld_names={e["name"] for e in json.loads(show(mb,"tests/fs2y_ledger.json")).get("entries",[])}
head_ld_names=set()
_ddir="tests/fs2y_ledger.d"
if os.path.isdir(_ddir):
    for _fn in glob.glob(os.path.join(_ddir,"*.json")):
        try:
            head_ld_names.add(json.load(open(_fn)).get("name"))
        except Exception:
            pass
else:
    head_ld_names={e["name"] for e in json.load(open("tests/fs2y_ledger.json")).get("entries",[])}

expected={
  "docs/model_facts.json": {
    "base_count": base_mf,
    "head_count": head_mf,
    "added": sorted(head_mf_keys - base_mf_keys),
    "removed": sorted(base_mf_keys - head_mf_keys),
  },
  "tests/fs2y_ledger.json": {
    "base_count": base_ld,
    "head_count": head_ld,
    "added": sorted(head_ld_names - base_ld_names),
    "removed": sorted(base_ld_names - head_ld_names),
  },
}

m=json.load(open("experiments/gpu-sim-034x/metrics.json"))
actual=m.get("protected_keys",{})
# support both old (int) and new (dict) form
mismatch=False
for f in expected:
    a=actual.get(f)
    if not isinstance(a, dict) or a != expected[f]:
        mismatch=True
        print(f"MISMATCH {f}: metrics={a} computed={expected[f]}")
if mismatch:
    print("FAIL: protected_keys in metrics.json do not match computed counts")
    exit(1)
print("OK: protected_keys match computed base/HEAD/added/removed counts")
PY

echo "run.sh complete"
