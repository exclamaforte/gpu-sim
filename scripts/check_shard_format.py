#!/usr/bin/env python3
"""Shard format checks for --fast (gpu-sim-0fr1). Pure file content, no build."""
import importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MDIR = ROOT / "scripts" / "fs2y_mutations.d"
VERDICT_PREFIXES = ("pass", "fail", "ok")


def check(root=None):
    """Return a list of error strings (empty == all shards valid)."""
    if root is None:
        root = ROOT
    root = Path(root)
    mdir = root / "scripts" / "fs2y_mutations.d"
    errors = []
    json_files = sorted(mdir.glob("*.json"))
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
        elif desc.lower().startswith(VERDICT_PREFIXES):
            errors.append(f"{key}: description reads like a verdict")
        # key == filename stem: the registry keys off the filename, so also
        # require the json "key" field to match the stem (no dead data).
        if isinstance(meta, dict) and "key" in meta and meta.get("key") != key:
            errors.append(f"{key}: json key field {meta.get('key')!r} != filename stem")
        py_path = mdir / (key + ".py")
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
        # missing .py requires data_only declaration (inherit k9e4 behaviour)
        if not py_path.exists():
            data_only = bool(meta.get("data_only", False)) or (
                "function" in meta and meta.get("function") is None)
            if not data_only:
                errors.append(f"{key}: missing .py and not declared data_only")
        keys.append(key)

    seen = set()
    for k in keys:
        if k in seen:
            errors.append(f"duplicate key: {k}")
        seen.add(k)

    return errors


def main():
    errors = check()
    if errors:
        for e in errors:
            print(e)
        sys.exit(1)
    json_files = sorted((ROOT / "scripts" / "fs2y_mutations.d").glob("*.json"))
    print(f"shard format: {len(json_files)} shards OK")


if __name__ == "__main__":
    sys.exit(main())
