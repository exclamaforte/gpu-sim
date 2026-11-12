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
    import json, subprocess, sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    # Pick a positive_only row with a mutation that is NOT in MUTATION_FUNCTIONS.
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
        r = subprocess.run(
            [sys.executable, "scripts/" + "check_append_only" + ".py"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=300,
        )
        out = (r.stdout or "") + (r.stderr or "")
        assert r.returncode != 0, "expected failure"
        assert "claims mutation_plant but has no registered mutation function" in out
    finally:
        shard.unlink(missing_ok=True)
