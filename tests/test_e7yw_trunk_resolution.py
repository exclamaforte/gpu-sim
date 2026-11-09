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
    # Replicate merge_base logic without importing the checker.
    import resolve_trunk
    trunk = resolve_trunk.resolve_trunk(str(ROOT)) or "main"
    r = subprocess.run(["git", "-C", str(ROOT), "merge-base", "HEAD", trunk],
                       capture_output=True, text=True)
    base = r.stdout.strip()
    integration_sha = trunk_sha(str(ROOT))
    assert base == integration_sha, (
        f"resolved base {base} != integration branch sha {integration_sha}")
