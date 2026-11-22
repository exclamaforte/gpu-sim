"""gpu-sim-cywp: the btrfs metadata headroom check must fail BEFORE rename() does.

Every case drives scripts/check_fs_headroom.py through --simulate, because the
interesting state (metadata block group full, device unallocated) is exactly the
state in which the box cannot be trusted to write the test output.  The numbers
are the ones measured during the outage:

    Metadata, DUP: total=13421772800  used=10665132032   (79.4%)
    Device unallocated: 1048576                          (1 MiB)
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "check_fs_headroom.py")

TOTAL = 13421772800      # 12.5 GiB, metadata DUP on this box
GIB = 1024 ** 3


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args],
                           capture_output=True, text=True)


def simulated(used_pct, unalloc):
    return run("--simulate", f"{int(TOTAL * used_pct / 100)},{TOTAL},{unalloc}")


def test_healthy_box_passes_and_says_df_is_not_authoritative():
    r = simulated(79.4, 1048576)                 # the measured post-recovery state
    assert r.returncode == 0, r.stdout + r.stderr
    assert "79.4%" in r.stdout
    assert "not authoritative" in r.stdout


def test_full_metadata_fails_even_though_data_is_free():
    r = simulated(98.0, 1048576)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FS METADATA HEADROOM: FAIL" in r.stdout
    assert "98.0%" in r.stdout
    # the message must be diagnosable, not just a number: name the mechanism
    # AND the remedies, because the error the agent actually saw ("no space
    # left on device" with 70 GiB free) sends them hunting the wrong thing.
    assert "rename()" in r.stdout
    assert "df reports free space" in r.stdout
    assert "reap_appendonly_worktrees.py" in r.stdout
    assert "git worktree prune" in r.stdout


def test_roomy_but_filling_metadata_fails_when_it_cannot_grow():
    """90% used with 0 unallocated is fatal-in-waiting: no chunk can be made."""
    r = simulated(91.0, 0)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "cannot be grown" in r.stdout


def test_same_fill_level_passes_when_the_device_can_grow():
    r = simulated(91.0, 2 * GIB)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "fs metadata headroom OK" in r.stdout


def test_unknown_unallocated_is_not_treated_as_zero():
    """A non-root btrfs-progs can omit the per-device block; 'unknown' must not
    fail the gate (it is not evidence of a problem)."""
    r = simulated(91.0, "-")
    assert r.returncode == 0, r.stdout + r.stderr


def test_threshold_is_configurable():
    r = run("--simulate", f"{int(TOTAL * 0.80)},{TOTAL},1048576", "--pct", "80")
    assert r.returncode == 1, r.stdout + r.stderr


def test_non_btrfs_path_is_a_skip_not_a_failure():
    """CI and other machines are not btrfs; the check must not invent a problem."""
    r = run("--path", "/proc")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "not btrfs" in r.stdout


def test_unreadable_path_is_a_note_not_a_failure(tmp_path):
    """If btrfs cannot be read at all, say so and stay out of the way."""
    r = run("--path", str(tmp_path / "does-not-exist"))
    assert r.returncode == 0, r.stdout + r.stderr


def test_gate_declares_the_stage_it_runs():
    """The gate must not 'declare' a stage it never runs (nfxi's manifest)."""
    gate = open(os.path.join(ROOT, "scripts", "gate.sh")).read()
    assert '"fs metadata headroom"' in gate
    assert 'run_hard "fs metadata headroom" python3 scripts/check_fs_headroom.py' in gate
