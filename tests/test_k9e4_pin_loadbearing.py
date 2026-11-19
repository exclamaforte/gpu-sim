"""Prove the k9e4 pinned base SHA is load-bearing (gpu-sim-mve4 arm 4)."""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _sandbox():
    parent = Path(tempfile.mkdtemp(dir=ROOT / ".lab" / "tmp", prefix="k9e4-pin-"))
    base = parent / "sandbox"
    # mkdtemp created parent; remove it so copytree can create base
    import shutil as _sh
    _sh.rmtree(parent)
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
        # Import the bound node's module.
        spec = importlib.util.spec_from_file_location(
            "k9e4_mod", ROOT / "tests" / "test_k9e4_registry_superset.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        bound_node = mod.test_branch_registry_is_superset_of_main

        # Force the pin to the advanced commit: the bound node must fail.
        mod.ROOT = base
        mod.PINNED_BASE_SHA = advanced_sha
        with pytest.raises(AssertionError) as exc:
            bound_node()
        assert "branch registry missing keys" in str(exc.value), exc.value

        # With the true pin: the bound node must pass.
        mod.PINNED_BASE_SHA = base_sha
        bound_node()  # must not raise
    finally:
        shutil.rmtree(base.parent, ignore_errors=True)
