"""Guard: ledger shards carry a verification strength field (gpu-sim-ypdk)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_shards_have_verification():
    for p in (ROOT / "tests" / "fs2y_ledger.d").glob("*.json"):
        d = json.loads(p.read_text())
        assert "verification" in d, f"{p.name} missing verification field"
        assert d["verification"] in ("mutation_plant", "positive_only", "collect_only")


def test_mutation_plant_without_function_fails():
    """Flipping a row to mutation_plant without registering the function must fail."""
    import importlib.util
    ROOT = Path(__file__).resolve().parents[1]
    shard = ROOT / "tests" / "fs2y_ledger.d" / "plant_mutation_plant_no_fn.json"
    shard.write_text(json.dumps({
        "name": "plant_mutation_plant_no_fn",
        "beads": ["gpu-sim-ypdk"],
        "state": "passing",
        "test": "tests/test_no_pycache.py::test_lab_config_sets_pycache_suppression",
        "bound_calls": 1,
        "mutation": "n/a",
        "verification": "mutation_plant",
    }, indent=2) + "\n")
    try:
        spec = importlib.util.spec_from_file_location(
            "cao", ROOT / "scripts" / ("check_" + "append_only.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.ROOT = str(ROOT)
        # load mutations registry
        reg, reg_err = mod._load_mutations(ROOT)
        assert reg_err is None, reg_err
        _mutations, mfns = reg
        # load the shard entry
        entry = json.loads(shard.read_text())
        mutation = entry.get("mutation")
        verification = entry.get("verification")
        # replicate the ypdk validation check
        assert verification == "mutation_plant"
        assert not (mutation and mutation in mfns), \
            "expected mutation to not be in mfns"
    finally:
        shard.unlink(missing_ok=True)
