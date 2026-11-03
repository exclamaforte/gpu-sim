"""Registry coverage for the fs2y ledger (gpu-sim-vb7x).

Every non-pending row in tests/fs2y_ledger.json names a mutation that must
resolve against scripts/fs2y_mutations.py. A missing registry file or a row
whose mutation is absent from it must fail closed (no silent pass). The
registry is a superset by design: extra names are allowed.
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
    assert REGISTRY.is_file() is True
    spec = importlib.util.spec_from_file_location("fs2y_mutations", REGISTRY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fs2y_mutations"] = mod
    spec.loader.exec_module(mod)
    mutations = getattr(mod, "MUTATIONS", None)
    assert isinstance(mutations, dict), (
        "scripts/fs2y_mutations.py: MUTATIONS missing or not a dict")
    return mutations


def _unregistered(ledger, mutations):
    """Return list of (row_name, mutation) pairs that are non-pending and
    unresolved against the registry."""
    out = []
    for entry in ledger.get("entries", []):
        if entry.get("state") == "pending":
            continue
        mutation = entry.get("mutation")
        if not mutation or mutation in ("n/a", "doc-only"):
            continue
        if mutation not in mutations:
            out.append((entry.get("name", "?"), mutation))
    return out


def test_registry_file_exists():
    assert REGISTRY.is_file() is True


def test_registry_imports():
    mutations = _load_mutations()
    assert len(mutations) > 0
    assert all(isinstance(k, str) and isinstance(v, str)
               for k, v in mutations.items()) is True



class TestFs2yMutations:
    def test_every_non_pending_mutation_registered(self):
        mutations = _load_mutations()
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        bad = _unregistered(ledger, mutations)
        assert len(bad) == 0, f"unregistered ledger mutations: {bad}"


def test_registered_mutation_passes():
    # Opposite direction: a scratch ledger row whose mutation IS registered
    # must not be reported as unresolved.
    mutations = _load_mutations()
    scratch = {"entries": [{"name": "scratch_ok", "state": "passing",
                            "mutation": next(iter(mutations))}]}
    assert len(_unregistered(scratch, mutations)) == 0


def test_unregistered_mutation_fails():
    # A scratch ledger row whose mutation is NOT registered must be reported.
    mutations = _load_mutations()
    scratch = {"entries": [{"name": "scratch_bad", "state": "passing",
                            "mutation": "definitely_not_a_registered_mutation"}]}
    bad = _unregistered(scratch, mutations)
    assert len(bad) == 1
    assert bad[0][0] == "scratch_bad"


def test_descriptions_are_short():
    mutations = _load_mutations()
    for name, desc in mutations.items():
        assert isinstance(desc, str) is True
        assert len(desc.strip()) > 0
        assert len(desc) <= 120, f"{name}: description too long"
        assert desc.lower().startswith(("pass", "fail", "ok")) is False, (
            f"{name}: description reads like a verdict")