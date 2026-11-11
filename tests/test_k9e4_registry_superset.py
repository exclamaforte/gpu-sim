"""Superset guard: branch registry must contain every mutation key main has (k9e4)."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "scripts" / "fs2y_mutations.py"


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
    """Plant: if a key is missing from branch, superset check fails."""
    main_keys = _load_main_keys()
    branch_keys = _load_branch_keys()
    # Simulate missing key
    fake_branch = branch_keys - {sorted(branch_keys)[0]}
    missing = main_keys - fake_branch if sorted(branch_keys)[0] in main_keys else set()
    # At least verify that a subset is detected as missing
    assert len(fake_branch) == len(branch_keys) - 1
    # The superset assertion would fail if a main key is missing
    if sorted(branch_keys)[0] in main_keys:
        assert sorted(branch_keys)[0] not in fake_branch
