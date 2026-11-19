"""Guard: checker resolves integration branch (main), not stale master (gpu-sim-e7yw)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from resolve_trunk import resolve_trunk, trunk_sha


def test_resolve_trunk_prefers_main():
    ref = resolve_trunk(str(ROOT))
    assert ref == "main", f"expected main, got {ref}"


def _independent_expected_sha():
    # Read the named ref's sha DIRECTLY, not through resolve_trunk, so the
    # comparison cannot be tautological.
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "main"], text=True).strip()


def test_resolved_base_equals_integration_branch():
    expected = _independent_expected_sha()
    actual = trunk_sha(str(ROOT))
    assert actual == expected, (
        f"trunk_sha {actual} != independently-read main sha {expected}")


def test_resolved_base_negative_arm():
    # Point the comparison at a different commit: it must NOT match.
    main_sha = _independent_expected_sha()
    other = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD~1"], text=True).strip()
    assert other != main_sha, "HEAD~1 must differ from main for this arm"
    assert trunk_sha(str(ROOT)) != other, (
        "trunk_sha must not equal an unrelated commit")
