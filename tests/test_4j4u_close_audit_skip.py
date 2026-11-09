"""Guard: close audit reports UNVERIFIABLE (rc=1) in a checkout with no beads DB (gpu-sim-4j4u rework)."""
import subprocess, os, tempfile, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def test_close_audit_skips_without_db():
    # True DB-less clone (real git clone, .beads removed): uses real bd.
    #
    # A UNIQUE scratch dir under .lab/tmp, not a fixed /var/tmp path: with a
    # fixed path the test rmtree'd another run's clone, so two concurrent runs
    # BOTH failed (FileNotFoundError / CalledProcessError), and because the
    # ledger row binds this test that collision made check_append_only -- which
    # every gate runs -- flake whenever two gates overlapped (gpu-sim-4j4u).
    # Cloning from REPO rather than a hardcoded absolute path keeps the test
    # independent of this machine's layout.
    (REPO / ".lab" / "tmp").mkdir(parents=True, exist_ok=True)
    dst = Path(tempfile.mkdtemp(prefix="realclone-4j4u-", dir=str(REPO / ".lab" / "tmp")))
    subprocess.run(["git", "clone", str(REPO), str(dst)], check=True, capture_output=True)
    beads = dst / ".beads"
    if beads.exists():
        shutil.rmtree(beads)
    # The allowlist ships in the repo, so a clone inherits it. Remove it here to
    # exercise the NO-ENTRY path, which must still fail (gpu-sim-4j4u).
    allow = dst / "tests" / "close_audit_allowlist.json"
    if allow.exists():
        allow.unlink()
    # Copy the modified script under test
    (dst / "scripts").mkdir(parents=True, exist_ok=True)
    import shutil as _sh
    _sh.copy(str(REPO / "scripts" / "close_requires_artifact.py"),
             str(dst / "scripts" / "close_requires_artifact.py"))
    try:
        r = subprocess.run(["python3", "scripts/close_requires_artifact.py"],
                           cwd=str(dst), capture_output=True, text=True)
        assert r.returncode != 0, f"expected UNVERIFIABLE rc!=0, got {r.returncode}: {r.stdout}\n{r.stderr}"
        assert "UNVERIFIABLE" in (r.stdout + r.stderr), f"expected UNVERIFIABLE message: {r.stdout}\n{r.stderr}"
    finally:
        shutil.rmtree(dst, ignore_errors=True)

def test_close_audit_audits_with_db():
    # In a worktree (DB available via parent), the audit runs and rc=0
    r = subprocess.run(["python3", "scripts/close_requires_artifact.py"],
                       cwd=str(REPO), capture_output=True, text=True)
    assert r.returncode == 0, f"expected audit rc=0, got {r.returncode}: {r.stdout}\n{r.stderr}"

def test_close_audit_allowlist_makes_the_skip_loud():
    """A tracked exemption converts the no-DB failure into a LOUD, attributed pass.

    Both halves matter: rc must become 0 (otherwise the exemption is decorative)
    and the output must name WHO and WHY on every use (otherwise it is a silent
    pass, which is the defect this bead exists to remove).
    """
    (REPO / ".lab" / "tmp").mkdir(parents=True, exist_ok=True)
    dst = Path(tempfile.mkdtemp(prefix="realclone-4j4u-", dir=str(REPO / ".lab" / "tmp")))
    subprocess.run(["git", "clone", str(REPO), str(dst)], check=True, capture_output=True)
    beads = dst / ".beads"
    if beads.exists():
        shutil.rmtree(beads)
    (dst / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(str(REPO / "scripts" / "close_requires_artifact.py"),
                str(dst / "scripts" / "close_requires_artifact.py"))
    try:
        r = subprocess.run(["python3", "scripts/close_requires_artifact.py"],
                           cwd=str(dst), capture_output=True, text=True)
        out = r.stdout + r.stderr
        assert r.returncode == 0, f"allowlisted run should pass, got {r.returncode}: {out}"
        assert "ALLOWLISTED: close audit skipped in this checkout" in out, (
            "the exemption was not printed loudly:\n" + out)
        assert "gpu-sim-pi" in out, "the exemption did not name WHO:\n" + out
    finally:
        shutil.rmtree(dst, ignore_errors=True)


def test_close_audit_allowlist_entry_without_who_is_ignored():
    """An entry that cannot name who/why must fail CLOSED, not exempt anything."""
    (REPO / ".lab" / "tmp").mkdir(parents=True, exist_ok=True)
    dst = Path(tempfile.mkdtemp(prefix="realclone-4j4u-", dir=str(REPO / ".lab" / "tmp")))
    subprocess.run(["git", "clone", str(REPO), str(dst)], check=True, capture_output=True)
    beads = dst / ".beads"
    if beads.exists():
        shutil.rmtree(beads)
    (dst / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(str(REPO / "scripts" / "close_requires_artifact.py"),
                str(dst / "scripts" / "close_requires_artifact.py"))
    try:
        (dst / "tests" / "close_audit_allowlist.json").write_text(
            '{"entries": [{"scope": "no-beads-db"}]}', encoding="utf-8")
        r = subprocess.run(["python3", "scripts/close_requires_artifact.py"],
                           cwd=str(dst), capture_output=True, text=True)
        out = r.stdout + r.stderr
        assert r.returncode != 0, f"an anonymous exemption must not pass: {out}"
        assert "UNVERIFIABLE" in out, f"expected UNVERIFIABLE: {out}"
        assert "ALLOWLISTED" not in out, f"an anonymous exemption must not be honoured: {out}"
    finally:
        shutil.rmtree(dst, ignore_errors=True)

def test_close_audit_allowlist_does_NOT_exempt_a_broken_db():
    """A present-but-unusable database must fail even when the allowlist exists.

    The allowlist covers one situation: this checkout has no database, so nobody
    could answer. A database that is LOCKED or CORRUPT is the failure the audit
    exists to report; exempting it would turn the single case worth catching into
    a green pass (gpu-sim-4j4u review, reviewer's fake-bd probe).
    """
    (REPO / ".lab" / "tmp").mkdir(parents=True, exist_ok=True)
    dst = Path(tempfile.mkdtemp(prefix="realclone-4j4u-", dir=str(REPO / ".lab" / "tmp")))
    fakebin = dst / "fakebin"
    try:
        subprocess.run(["git", "clone", str(REPO), str(dst / "repo")],
                       check=True, capture_output=True)
        repo = dst / "repo"
        beads = repo / ".beads"
        if beads.exists():
            shutil.rmtree(beads)
        # Run the WORKING-TREE script, not the clone's committed copy.
        (repo / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copy(str(REPO / "scripts" / "close_requires_artifact.py"),
                    str(repo / "scripts" / "close_requires_artifact.py"))
        # A bd that EXISTS and FAILS on a present database, never "absent".
        fakebin.mkdir(parents=True, exist_ok=True)
        fake = fakebin / "bd"
        fake.write_text("#!/bin/sh\necho 'dolt: database is locked' >&2\nexit 1\n")
        fake.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = str(fakebin) + os.pathsep + env.get("PATH", "")
        r = subprocess.run(["python3", "scripts/close_requires_artifact.py"],
                           cwd=str(repo), capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        # The allowlist is present in this clone (it ships in the repo).
        assert (repo / "tests" / "close_audit_allowlist.json").exists(), "test needs the allowlist present"
        assert r.returncode != 0, f"a locked DB must NOT be exempted: {out}"
        assert "UNVERIFIABLE" in out, f"expected UNVERIFIABLE: {out}"
        assert "ALLOWLISTED" not in out, f"the allowlist must not cover a broken DB: {out}"
        assert "locked" in out.lower(), f"the real cause should be reported: {out}"
    finally:
        shutil.rmtree(dst, ignore_errors=True)

def test_close_audit_allowlist_does_NOT_exempt_a_missing_engine():
    """bd present but its engine missing is an ENVIRONMENT failure, not absence.

    A fake bd that exits 127 with 'dolt: command not found' used to classify as
    no_database (via a 'dolt ... not found' heuristic), so an allowlisted box
    exempted EVERY checkout on it -- including ones that really do have a
    database -- while printing a reason ('this checkout has no .beads directory')
    that was false for the case that fired (gpu-sim-4j4u review, round two).
    """
    (REPO / ".lab" / "tmp").mkdir(parents=True, exist_ok=True)
    dst = Path(tempfile.mkdtemp(prefix="realclone-4j4u-", dir=str(REPO / ".lab" / "tmp")))
    fakebin = dst / "fakebin"
    try:
        subprocess.run(["git", "clone", str(REPO), str(dst / "repo")],
                       check=True, capture_output=True)
        repo = dst / "repo"
        (repo / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copy(str(REPO / "scripts" / "close_requires_artifact.py"),
                    str(repo / "scripts" / "close_requires_artifact.py"))
        fakebin.mkdir(parents=True, exist_ok=True)
        fake = fakebin / "bd"
        fake.write_text("#!/bin/sh\necho 'dolt: command not found' >&2\nexit 127\n")
        fake.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = str(fakebin) + os.pathsep + env.get("PATH", "")
        r = subprocess.run(["python3", "scripts/close_requires_artifact.py"],
                           cwd=str(repo), capture_output=True, text=True, env=env)
        out = r.stdout + r.stderr
        assert (repo / "tests" / "close_audit_allowlist.json").exists(), "allowlist must be present"
        assert r.returncode != 0, f"a missing engine must NOT be exempted: {out}"
        assert "UNVERIFIABLE" in out, f"expected UNVERIFIABLE: {out}"
        assert "ALLOWLISTED" not in out, f"a missing engine must not be allowlisted: {out}"
    finally:
        shutil.rmtree(dst, ignore_errors=True)
