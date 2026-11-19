"""Guard: generate_inventory resolves base ref and fails loudly (gpu-sim-vmq2)."""
import hashlib, json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN_REL = "experiments/gpu-sim-u2o.6.11/generate_inventory.py"
INV_REL = "experiments/gpu-sim-u2o.6.11/tracks_inventory.md"
FIXED_BASE = "88f2e809"


def test_bad_base_exits_nonzero():
    r = subprocess.run(
        [sys.executable, str(ROOT / GEN_REL), "--base", "nonexistent-ref-xyz"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)


def test_replay_unchanged_in_clone():
    # no-arg replay is NOT what we test; instead, replay against a FIXED
    # earlier base in a disposable clone of HEAD and assert byte-identical
    # to the committed file.  Perturbing the SHA in the committed file
    # must make this fail.
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "clone"
        subprocess.run(["git", "clone", "-q", str(ROOT), str(clone)], check=True)
        inv = clone / INV_REL
        committed = (ROOT / INV_REL).read_bytes()
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(committed)
        # use the worktree generator (updated version)
        import shutil
        shutil.copyfile(ROOT / GEN_REL, clone / GEN_REL)
        before = hashlib.sha256(committed).hexdigest()
        r = subprocess.run(
            [sys.executable, GEN_REL, "--base", FIXED_BASE],
            cwd=str(clone), capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        after = hashlib.sha256(inv.read_bytes()).hexdigest()
        assert before == after, "regenerated file differs from committed"
