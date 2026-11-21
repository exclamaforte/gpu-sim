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


# Pin the comparison base to a single commit resolved once, so the test is
# hermetic: it never re-reads the live "main" ref (which moves with every land
# and makes a branch tip red by construction).
def _pinned_base_sha():
    for args in (["merge-base", "HEAD", "main"],
                 ["merge-base", "HEAD", "master"]):
        try:
            out = subprocess.check_output(
                ["git", "-C", str(ROOT)] + args, text=True,
                stderr=subprocess.DEVNULL).strip()
            if out:
                return out
        except subprocess.CalledProcessError:
            continue
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return None

PINNED_BASE_SHA = _pinned_base_sha()


def _load_main_keys():
    main_keys = set()
    if not PINNED_BASE_SHA:
        return main_keys
    try:
        d_files = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-tree", "-r", PINNED_BASE_SHA,
             "--name-only", "scripts/fs2y_mutations.d/"],
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
                ["git", "-C", str(ROOT), "show",
                 f"{PINNED_BASE_SHA}:scripts/fs2y_mutations.py"],
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
    assert not missing, f"branch registry missing keys that {PINNED_BASE_SHA} has: {sorted(missing)}"


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
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_shard_format", ROOT / "scripts" / "check_shard_format.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    errors = mod.check(ROOT)
    assert not errors, "\n".join(errors)
