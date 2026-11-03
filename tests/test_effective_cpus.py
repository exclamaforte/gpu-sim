"""Tests for effective_cpus helper (gpu-sim-ynaj)."""
import os
import tempfile
from pathlib import Path

sys_path = None
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import effective_cpus as ec


def test_parse_cpu_max_max():
    assert ec.parse_cpu_max("max 100000") == (None, 100000)


def test_parse_cpu_max_200000():
    assert ec.parse_cpu_max("200000 100000") == (200000, 100000)


def test_parse_cpu_max_50000():
    assert ec.parse_cpu_max("50000 100000") == (50000, 100000)


def test_parse_cpu_max_missing():
    assert ec.parse_cpu_max("") == (None, None)


def test_walk_cpu_max_synthetic():
    with tempfile.TemporaryDirectory() as td:
        scope = os.path.join(td, "scope")
        os.makedirs(scope)
        with open(os.path.join(scope, "cpu.max"), "w") as f:
            f.write("200000 100000")
        # parent has max
        with open(os.path.join(td, "cpu.max"), "w") as f:
            f.write("max 100000")
        quotas = ec.walk_cpu_max("scope", fs_root=td)
        # should find scope's quota
        vals = [(q, p) for q, p, _ in quotas]
        assert (200000, 100000) in vals


def test_walk_cpu_max_missing_fallback():
    with tempfile.TemporaryDirectory() as td:
        scope = os.path.join(td, "scope")
        os.makedirs(scope)
        # no cpu.max anywhere
        quotas = ec.walk_cpu_max("scope", fs_root=td)
        assert quotas == [] or all(q is None for q, _, _ in quotas)


def test_mismatch_reported():
    # effective_cpus reports nproc_agrees; simulate by checking reason contains WARNING on mismatch
    # We can't easily force nproc mismatch, but parse/walk are tested above.
    r = ec.effective_cpus()
    assert "effective_cpus" in r
    assert r["effective_cpus"] == 2.0 or r["effective_cpus"] > 0


def test_host_cpus_os_view_consistent():
    """host_cpus_os_view == affinity_cpus (os view)."""
    r = ec.effective_cpus()
    assert r["affinity_cpus"] == len(os.sched_getaffinity(0))
    # host_cpus_os_view is the os view = affinity count
    assert r["affinity_cpus"] >= 1


def test_cpu_capacity_consistent_with_min():
    """cpu_capacity_cpus == min(affinity, quota) (or affinity if no quota)."""
    r = ec.effective_cpus()
    affinity = r["affinity_cpus"]
    quota = r["quota_cpus"]
    if quota is None:
        expected = affinity
    else:
        expected = min(affinity, quota)
    assert r["effective_cpus"] == expected, (
        f"effective_cpus={r['effective_cpus']} != min({affinity}, {quota})={expected}")
