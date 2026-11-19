"""Prove the k9e4 pinned base SHA is load-bearing (gpu-sim-mve4 arm 4)."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sandbox():
    base = ROOT / ".lab" / "tmp" / "k9e4-pin-sandbox"
    if base.exists():
        shutil.rmtree(base)
    base.parent.mkdir(parents=True, exist_ok=True)

    def ignore(src, names):
        return {n for n in names if n in {".git", ".lab", "__pycache__"}}
    shutil.copytree(ROOT, base, ignore=ignore)
    subprocess.run(["git", "-C", str(base), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(base), "config", "user.email", "t@t.com"], check=True)
    subprocess.run(["git", "-C", str(base), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(base), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(base), "commit", "-q", "-m", "base"], check=True)
    base_sha = subprocess.check_output(
        ["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip()

    new_key = "zzzz_pin_loadbearing"
    shard_dir = base / "scripts" / "fs2y_mutations.d"
    (shard_dir / (new_key + ".json")).write_text(
        json.dumps({"description": "pin loadbearing probe"}))
    (shard_dir / (new_key + ".py")).write_text(
        'MUTATIONS = {"%s": "pin loadbearing probe"}\n' % new_key)
    subprocess.run(["git", "-C", str(base), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(base), "commit", "-q", "-m", "advance"], check=True)
    advanced_sha = subprocess.check_output(
        ["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip()
    return base, base_sha, advanced_sha, new_key


def test_pin_is_loadbearing():
    base, base_sha, advanced_sha, new_key = _sandbox()
    try:
        # Load the k9e4 test module from the real repo.
        spec = importlib.util.spec_from_file_location(
            "k9e4_mod", ROOT / "tests" / "test_k9e4_registry_superset.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Branch keys (real repo) do not contain new_key.
        branch_keys = mod._load_branch_keys()

        # Force the pin to the advanced commit: new_key must surface as missing.
        mod.ROOT = base
        mod.PINNED_BASE_SHA = advanced_sha
        main_keys = mod._load_main_keys()
        missing = main_keys - branch_keys
        assert new_key in missing, (
            "pin not load-bearing: forced advanced pin did not surface the new key")

        # With the pin at the base commit, new_key must not be missing.
        mod.PINNED_BASE_SHA = base_sha
        main_keys2 = mod._load_main_keys()
        missing2 = main_keys2 - branch_keys
        assert new_key not in missing2, (
            "base pin unexpectedly includes the advanced-only key")
    finally:
        shutil.rmtree(base, ignore_errors=True)
