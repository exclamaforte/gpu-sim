"""Guard: generate_inventory resolves base ref and fails loudly (gpu-sim-vmq2)."""
import hashlib, json, subprocess, sys, tempfile, re
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


def test_replay_preserves_sha():
    # Replay against a FIXED earlier base in a disposable clone and assert the
    # recorded base sha is preserved (a timestamp is stamped, so byte-identical
    # is no longer required, but the sha must not drift).
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "clone"
        subprocess.run(["git", "clone", "-q", str(ROOT), str(clone)], check=True)
        inv = clone / INV_REL
        committed = (ROOT / INV_REL).read_bytes()
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(committed)
        import shutil
        shutil.copyfile(ROOT / GEN_REL, clone / GEN_REL)
        r = subprocess.run(
            [sys.executable, GEN_REL, "--base", FIXED_BASE],
            cwd=str(clone), capture_output=True, text=True, timeout=120,
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        after = inv.read_text()
        assert FIXED_BASE in after, "recorded base sha drifted on replay"


def test_no_arg_moved_pointer_fails():
    # NO-ARG arm against a moved pointer: clone whose origin/main has advanced
    # past the note's sha. Run with no --base; expect nonzero naming both shas.
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "clone"
        subprocess.run(["git", "clone", "-q", str(ROOT), str(clone)], check=True)
        inv = clone / INV_REL
        committed = (ROOT / INV_REL).read_bytes()
        inv.parent.mkdir(parents=True, exist_ok=True)
        inv.write_bytes(committed)
        import shutil
        shutil.copyfile(ROOT / GEN_REL, clone / GEN_REL)
        r = subprocess.run(
            [sys.executable, GEN_REL],
            cwd=str(clone), capture_output=True, text=True, timeout=120,
        )
        assert r.returncode != 0, (r.returncode, r.stdout, r.stderr)
        msg = (r.stdout + r.stderr)
        assert FIXED_BASE in msg, f"recorded sha {FIXED_BASE} not in message: {msg}"
        # resolved sha is the clone's origin/main tip (e1a2aaaa prefix or main tip)
        assert "resolved" in msg.lower() or "mismatch" in msg.lower(), msg