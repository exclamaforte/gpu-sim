"""Guard: generate_inventory resolves base ref and fails loudly (gpu-sim-vmq2)."""
import hashlib, json, subprocess, sys, tempfile, re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN_REL = "experiments/gpu-sim-u2o.6.11/generate_inventory.py"
INV_REL = "experiments/gpu-sim-u2o.6.11/tracks_inventory.md"
FIXED_BASE = "e1a2aaaa"   # current main tip the committed note names


def _clone_without_local_master(root, dest):
    subprocess.run(["git", "clone", "-q", str(root), str(dest)], check=True)
    for b in ("master", "main"):
        subprocess.run(["git", "-C", str(dest), "branch", "-D", b],
                       capture_output=True, text=True)
    # keep origin/main pointing where the clone's HEAD resolves; ensure it
    # exists (silent no-op on a fresh clone is a real failure mode)
    assert (dest / ".git").exists()


def test_bad_base_exits_nonzero():
    inv = ROOT / INV_REL
    before = hashlib.md5(inv.read_bytes()).hexdigest()
    r = subprocess.run(
        [sys.executable, str(ROOT / GEN_REL), "--base", "nonexistent-ref-xyz"],
        capture_output=True, text=True, timeout=60,
    )
    after = hashlib.md5(inv.read_bytes()).hexdigest()
    assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)
    assert before == after, "bad-base run must not touch the artifact"
    assert "nonexistent-ref-xyz" in (r.stdout + r.stderr)


def test_default_clone_byte_identical():
    # No-arg run in a clone whose origin resolves to the same commit the
    # committed note names: must succeed and leave the artifact byte-identical.
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "clone"
        _clone_without_local_master(ROOT, clone)
        inv = clone / INV_REL
        committed = (ROOT / INV_REL).read_bytes()
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(committed)
        shutil.copyfile(ROOT / GEN_REL, clone / GEN_REL)
        # also ship resolve_trunk helper
        (clone / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / "resolve_trunk.py",
                        clone / "scripts" / "resolve_trunk.py")
        r = subprocess.run(
            [sys.executable, GEN_REL],
            cwd=str(clone), capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert inv.read_bytes() == committed, "default run must be byte-identical"


def test_default_clone_moved_pointer_fails():
    # No-arg run where the resolved ref points at an older commit: must exit
    # nonzero and name both the recorded and the resolved sha.
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "clone"
        _clone_without_local_master(ROOT, clone)
        inv = clone / INV_REL
        committed = (ROOT / INV_REL).read_bytes()
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(committed)
        shutil.copyfile(ROOT / GEN_REL, clone / GEN_REL)
        (clone / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / "resolve_trunk.py",
                        clone / "scripts" / "resolve_trunk.py")
        # move the remote pointer one commit back
        subprocess.run(
            ["git", "-C", str(clone), "update-ref", "refs/remotes/origin/main",
             "HEAD~1"],
            check=True,
        )
        r = subprocess.run(
            [sys.executable, GEN_REL],
            cwd=str(clone), capture_output=True, text=True, timeout=120,
        )
        assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)
        msg = r.stdout + r.stderr
        assert FIXED_BASE in msg, f"recorded sha not in message: {msg}"


def test_replay_preserves_sha():
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "clone"
        _clone_without_local_master(ROOT, clone)
        inv = clone / INV_REL
        committed = (ROOT / INV_REL).read_bytes()
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(committed)
        shutil.copyfile(ROOT / GEN_REL, clone / GEN_REL)
        (clone / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / "resolve_trunk.py",
                        clone / "scripts" / "resolve_trunk.py")
        r = subprocess.run(
            [sys.executable, GEN_REL, "--base", FIXED_BASE],
            cwd=str(clone), capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        after = inv.read_text()
        assert FIXED_BASE in after, "recorded base sha drifted on replay"
