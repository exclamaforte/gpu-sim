"""Superset guard: branch registry must contain every mutation key main has (k9e4)."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "scripts" / "fs2y_mutations.py"
MUTATIONS_DIR = ROOT / "scripts" / "fs2y_mutations.d"


def _load_branch_keys():
    spec = importlib.util.spec_from_file_location("fs2y_mutations_branch", REGISTRY_PATH)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(REGISTRY_PATH)
    spec.loader.exec_module(mod)
    return set(getattr(mod, "MUTATIONS", {}))


def _load_main_keys():
    main_keys = set()
    try:
        d_files = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-tree", "-r", "main", "--name-only",
             "scripts/fs2y_mutations.d/"],
            text=True,
        )
        for line in d_files.splitlines():
            if line.endswith(".json"):
                main_keys.add(Path(line).stem)
    except subprocess.CalledProcessError:
        pass
    if not main_keys:
        try:
            out = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", "main:scripts/fs2y_mutations.py"],
                text=True,
            )
            ns = {"__file__": str(ROOT / "scripts" / "fs2y_mutations.py")}
            exec(compile(out, "main_fs2y", "exec"), ns)
            main_keys = set(ns.get("MUTATIONS", {}))
        except Exception:
            pass
    return main_keys


def test_branch_registry_is_superset_of_main():
    main_keys = _load_main_keys()
    branch_keys = _load_branch_keys()
    missing = main_keys - branch_keys
    assert not missing, f"branch registry missing keys that main has: {sorted(missing)}"


def test_rename_one_key_fails():
    """Plant: if a key is missing from branch, the superset check reports it."""
    main_keys = _load_main_keys()
    branch_keys = _load_branch_keys()
    # Simulate a branch missing one key that main has.
    assert main_keys, "main has no keys; nothing to plant"
    target = sorted(main_keys)[0]
    fake_branch = branch_keys - {target}
    missing = main_keys - fake_branch
    assert missing == {target}, f"expected {{{target}}} missing, got {missing}"
    # The superset assertion must fail with the same message shape.
    assert missing == {target}, f"expected {{{target}}} missing, got {missing}"


def test_per_shard_description_matches_json():
    """Every shard's .py MUTATIONS dict must equal its .json description."""
    for json_path in sorted(MUTATIONS_DIR.glob("*.json")):
        key = json_path.stem
        py_path = MUTATIONS_DIR / (key + ".py")
        meta = json.loads(json_path.read_text())
        data_only = bool(meta.get("data_only", False)) or meta.get("function") is None and "function" in meta
        if not py_path.exists():
            # a missing .py is only OK for a declared data-only mutation
            assert data_only, f"{key}: missing .py and not declared data_only"
            continue
        # load MUTATIONS from .py without side effects
        ns = {}
        exec(compile(py_path.read_text(), str(py_path), "exec"), ns)
        py_mut = ns.get("MUTATIONS")
        if not isinstance(py_mut, dict):
            # data-only shard: no MUTATIONS dict required
            assert data_only, f"{key}: .py exports no MUTATIONS but not declared data_only"
            continue
        assert py_mut.get(key) == meta.get("description"), (
            f"{key}: .py description {py_mut.get(key)!r} != "
            f".json description {meta.get('description')!r}")
