#!/usr/bin/env python3
"""Append-only guard for accumulating records.

Compares HEAD against the merge-base with master for two record files:
  - docs/model_facts.json: top-level fact-block keys
  - tests/fs2y_ledger.json: entry 'name' ids
Any key/entry present at the base but absent at HEAD fails unless it is named
in an explicit retirement list inside the same file, where each retirement
entry must carry a 'reason' (str) and 'commit' (str).
"""
import json, subprocess, sys, os, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def git_show(ref, path):
    r = subprocess.run(["git", "-C", ROOT, "show", f"{ref}:{path}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return r.stdout

def merge_base():
    r = subprocess.run(["git", "-C", ROOT, "merge-base", "HEAD", "master"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # fall back to HEAD~1? No; require merge-base
        return None
    return r.stdout.strip()

def retired_set_model_facts(d):
    out = set()
    for r in d.get("_retired_fact_blocks", []) or []:
        if isinstance(r, dict) and "key" in r:
            # validate reason + commit
            if not isinstance(r.get("reason"), str) or not r.get("reason"):
                raise ValueError(f"retirement entry for {r['key']} missing reason")
            if not isinstance(r.get("commit"), str) or not r.get("commit"):
                raise ValueError(f"retirement entry for {r['key']} missing commit")
            out.add(r["key"])
    return out

def retired_set_ledger(d):
    out = set()
    for r in d.get("retired", []) or []:
        if isinstance(r, dict) and "name" in r:
            if not isinstance(r.get("reason"), str) or not r.get("reason"):
                raise ValueError(f"retirement entry for {r['name']} missing reason")
            if not isinstance(r.get("commit"), str) or not r.get("commit"):
                raise ValueError(f"retirement entry for {r['name']} missing commit")
            out.add(r["name"])
    return out

def check_model_facts(base_ref):
    base_text = git_show(base_ref, "docs/model_facts.json")
    if base_text is None:
        return []  # no base file -> nothing to protect
    base = json.loads(base_text)
    head = json.load(open(os.path.join(ROOT, "docs", "model_facts.json")))
    base_keys = set(base.keys()) - {"_retired_fact_blocks"}
    head_keys = set(head.keys()) - {"_retired_fact_blocks"}
    retired = retired_set_model_facts(head)
    lost = base_keys - head_keys - retired
    return sorted(lost)

def check_ledger(base_ref):
    base_text = git_show(base_ref, "tests/fs2y_ledger.json")
    if base_text is None:
        return []
    base = json.loads(base_text)
    head = json.load(open(os.path.join(ROOT, "tests", "fs2y_ledger.json")))
    base_ids = {e.get("name") for e in base.get("entries", []) if e.get("name")}
    head_ids = {e.get("name") for e in head.get("entries", []) if e.get("name")}
    retired = retired_set_ledger(head)
    lost = base_ids - head_ids - retired
    return sorted(lost)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="base ref (default: merge-base HEAD master)")
    args = ap.parse_args()
    base_ref = args.base if args.base else merge_base()
    if base_ref is None:
        print("FAIL: cannot determine merge-base with master")
        return 1
    errors = []
    try:
        lost = check_model_facts(base_ref)
        for k in lost:
            errors.append(f"docs/model_facts.json lost fact block '{k}' (not retired)")
    except Exception as e:
        errors.append(f"docs/model_facts.json: {e}")
    try:
        lost = check_ledger(base_ref)
        for k in lost:
            errors.append(f"tests/fs2y_ledger.json lost ledger entry '{k}' (not retired)")
    except Exception as e:
        errors.append(f"tests/fs2y_ledger.json: {e}")
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("OK: append-only check passed (no unretired losses)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
