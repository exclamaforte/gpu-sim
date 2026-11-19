"""Guard: generate_inventory resolves base ref and fails loudly (gpu-sim-vmq2)."""
import hashlib, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN_REL = "experiments/gpu-sim-u2o.6.11/generate_inventory.py"
INV_REL = "experiments/gpu-sim-u2o.6.11/tracks_inventory.md"


def _committed_note_sha():
    text = (ROOT / INV_REL).read_text()
    m = re.search(r"base: ([0-9a-fA-F]{7,40})", text)
    assert m, "committed note has no base sha"
    return m.group(1)


def _clone_without_local_master(root, dest):
    subprocess.run(["git", "clone", "-q", str(root), str(dest)], check=True)
    for b in ("master", "main"):
        subprocess.run(["git", "-C", str(dest), "branch", "-D", b],
                       capture_output=True, text=True)


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
    note_sha = _committed_note_sha()
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "clone"
        _clone_without_local_master(ROOT, clone)
        # Pin the remote to the exact commit the committed note names.
        subprocess.run(
            ["git", "-C", str(clone), "update-ref", "refs/remotes/origin/main",
             note_sha],
            check=True,
        )
        inv = clone / INV_REL
        committed = (ROOT / INV_REL).read_bytes()
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(committed)
        shutil.copyfile(ROOT / GEN_REL, clone / GEN_REL)
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
    note_sha = _committed_note_sha()
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "clone"
        _clone_without_local_master(ROOT, clone)
        # Move the remote one commit behind the committed note.
        subprocess.run(
            ["git", "-C", str(clone), "update-ref", "refs/remotes/origin/main",
             note_sha + "~1"],
            check=True,
        )
        inv = clone / INV_REL
        committed = (ROOT / INV_REL).read_bytes()
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(committed)
        shutil.copyfile(ROOT / GEN_REL, clone / GEN_REL)
        (clone / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / "resolve_trunk.py",
                        clone / "scripts" / "resolve_trunk.py")
        r = subprocess.run(
            [sys.executable, GEN_REL],
            cwd=str(clone), capture_output=True, text=True, timeout=120,
        )
        assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)
        msg = r.stdout + r.stderr
        assert note_sha[:8] in msg or note_sha in msg, f"recorded sha not in message: {msg}"
        # resolved sha is the parent commit
        resolved = subprocess.check_output(
            ["git", "-C", str(clone), "rev-parse", "origin/main"],
            text=True).strip()
        assert resolved[:8] in msg or resolved in msg, f"resolved sha not in message: {msg}"


def test_replay_preserves_sha():
    note_sha = _committed_note_sha()
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "clone"
        _clone_without_local_master(ROOT, clone)
        subprocess.run(
            ["git", "-C", str(clone), "update-ref", "refs/remotes/origin/main",
             note_sha],
            check=True,
        )
        inv = clone / INV_REL
        committed = (ROOT / INV_REL).read_bytes()
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(committed)
        shutil.copyfile(ROOT / GEN_REL, clone / GEN_REL)
        (clone / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / "resolve_trunk.py",
                        clone / "scripts" / "resolve_trunk.py")
        r = subprocess.run(
            [sys.executable, GEN_REL, "--base", note_sha[:8]],
            cwd=str(clone), capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        after = inv.read_text()
        assert note_sha[:8] in after, "recorded base sha drifted on replay"
