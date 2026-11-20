#!/usr/bin/env python3
"""Shard format checks for --fast (gpu-sim-0fr1). Pure file content, no build."""
import importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MDIR = ROOT / "scripts" / "fs2y_mutations.d"
errors = []

json_files = sorted(MDIR.glob("*.json"))
keys = []
for jp in json_files:
    key = jp.stem
    try:
        meta = json.loads(jp.read_text())
    except Exception as e:
        errors.append(f"{key}: invalid json: {e}")
        continue
    desc = meta.get("description")
    if not isinstance(desc, str) or not desc.strip():
        errors.append(f"{key}: empty description")
    elif len(desc) > 120:
        errors.append(f"{key}: description too long ({len(desc)} > 120)")
    elif desc.lower().startswith(("pass", "fail", "ok")):
        errors.append(f"{key}: description reads like a verdict")
    # key == filename stem already by construction
    py_path = MDIR / (key + ".py")
    if py_path.exists():
        ns = {}
        try:
            exec(compile(py_path.read_text(), str(py_path), "exec"), ns)
        except Exception as e:
            errors.append(f"{key}: .py exec error: {e}")
            continue
        py_mut = ns.get("MUTATIONS")
        if isinstance(py_mut, dict):
            if py_mut.get(key) != desc:
                errors.append(f"{key}: .py description != .json description")
    keys.append(key)

# duplicate keys
seen = set()
for k in keys:
    if k in seen:
        errors.append(f"duplicate key: {k}")
    seen.add(k)

if errors:
    for e in errors:
        print(e)
    sys.exit(1)
print(f"shard format: {len(keys)} shards OK")
