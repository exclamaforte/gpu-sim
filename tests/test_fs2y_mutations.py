"""Registry coverage for the fs2y ledger (gpu-sim-vb7x).

Every non-pending row in tests/fs2y_ledger.json names a mutation that must
resolve against scripts/fs2y_mutations.py. A missing registry file or a row
whose mutation is absent from it must fail closed (no silent pass).
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "tests" / "fs2y_ledger.json"
REGISTRY = ROOT / "scripts" / "fs2y_mutations.py"


def _load_mutations():
    # Fail closed: a missing/unimportable registry is a failure, not a skip.
    assert REGISTRY.is_file(), f"registry missing: {REGISTRY}"
    spec = importlib.util.spec_from_file_location("fs2y_mutations", REGISTRY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fs2y_mutations"] = mod
    spec.loader.exec_module(mod)
    mutations = getattr(mod, "MUTATIONS", None)
    assert isinstance(mutations, dict), (
        "scripts/fs2y_mutations.py: MUTATIONS missing or not a dict")
    return mutations


def test_registry_file_exists():
    assert REGISTRY.is_file(), f"registry missing: {REGISTRY}"


def test_registry_imports():
    mutations = _load_mutations()
    assert isinstance(mutations, dict)
    assert all(isinstance(k, str) and isinstance(v, str)
               for k, v in mutations.items())


def test_every_non_pending_mutation_registered():
    mutations = _load_mutations()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    for entry in ledger.get("entries", []):
        state = entry.get("state", "passing")
        if state == "pending":
            continue
        mutation = entry.get("mutation")
        if not mutation or mutation in ("n/a", "doc-only"):
            continue
        name = entry.get("name", "?")
        assert mutation in mutations, (
            f"ledger row {name!r}: mutation {mutation!r} not in registry")


def test_descriptions_are_short():
    mutations = _load_mutations()
    for name, desc in mutations.items():
        assert isinstance(desc, str) and desc.strip()
        assert len(desc) <= 120, f"{name}: description too long"
        # descriptions document intent, not verdicts
        assert not desc.lower().startswith(("pass", "fail", "ok")), (
            f"{name}: description reads like a verdict")
