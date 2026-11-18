"""Guard: ledger shards carry a verification strength field (gpu-sim-ypdk)."""
import json, os, shutil, tempfile
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

    # Build a throwaway root so we don't mutate tracked files and don't run
    # the full 4-minute walk.
    with tempfile.TemporaryDirectory() as td:
        troot = Path(td)
        (troot / "tests" / "fs2y_ledger.d").mkdir(parents=True)
        (troot / "scripts" / "fs2y_mutations.d").mkdir(parents=True)
        # copy mutations registry
        shutil.copy(ROOT / "scripts" / "fs2y_mutations.py",
                    troot / "scripts" / "fs2y_mutations.py")
        # copy mutations dir contents (so _load_mutations works)
        for f in (ROOT / "scripts" / "fs2y_mutations.d").iterdir():
            if f.is_file():
                shutil.copy(f, troot / "scripts" / "fs2y_mutations.d" / f.name)
        # write the bad shard
        shard = troot / "tests" / "fs2y_ledger.d" / "plant_mutation_plant_no_fn.json"
        shard.write_text(json.dumps({
            "name": "plant_mutation_plant_no_fn",
            "beads": ["gpu-sim-ypdk"],
            "state": "passing",
            "test": "tests/test_no_pycache.py::test_lab_config_sets_pycache_suppression",
            "bound_calls": 1,
            "mutation": "ypdk_named_no_fn",
            "verification": "mutation_plant",
        }, indent=2) + "\n")
        # load checker against troot
        spec = importlib.util.spec_from_file_location(
            "cao", ROOT / "scripts" / ("check_" + "append_only.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.ROOT = str(troot)
        errs = mod.check_ledger_test_ids(troot)
        assert any("claims mutation_plant but has no registered mutation function" in e
                   for e in errs), errs
