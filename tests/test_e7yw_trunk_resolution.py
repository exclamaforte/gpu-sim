"""Guard: checker resolves integration branch (main), not stale master (gpu-sim-e7yw)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from resolve_trunk import resolve_trunk, trunk_sha


def test_resolve_trunk_prefers_main():
    ref = resolve_trunk(str(ROOT))
    assert ref == "main", f"expected main, got {ref}"


def test_resolved_base_equals_integration_branch():
    import sys, os, subprocess
    sys.path.insert(0, str(ROOT / "scripts"))
    import resolve_trunk
    trunk = resolve_trunk.resolve_trunk(str(ROOT)) or "main"
    # trunk_sha must agree with the ref resolve_trunk actually names.
    expected_sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", trunk], text=True).strip()
    actual_sha = resolve_trunk.trunk_sha(str(ROOT))
    assert actual_sha == expected_sha, (
        f"trunk_sha {actual_sha} != {trunk} sha {expected_sha}")
    # merge-base with that trunk must exist (well-defined base).
    r = subprocess.run(["git", "-C", str(ROOT), "merge-base", "HEAD", trunk],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"no merge-base between HEAD and {trunk}: {r.stderr}"
    # Only when the tree itself is on main must resolution equal HEAD's sha.
    try:
        head_ref = subprocess.check_output(
            ["git", "-C", str(ROOT), "symbolic-ref", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        head_ref = ""
    if head_ref == "main":
        head_sha = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        assert actual_sha == head_sha, (
            f"on main, resolved base {actual_sha} != HEAD sha {head_sha}")
