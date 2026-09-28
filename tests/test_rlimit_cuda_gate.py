#!/usr/bin/env python3
"""CPU-only invariants for the rlimit CUDA-compatibility gate (6.3.1).

No GPU and no network: every check here is a property of code and numbers.
Criterion 3's injected-failure test is in this file too, so the diagnostic
path is demonstrated without a device.

Run:  python3 tests/test_rlimit_cuda_gate.py
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from epic_u2o.rlimit_cuda_gate import (  # noqa: E402
    LIMIT_TABLE, PROBE_SOURCE, RlimitReintroductionError,
    assert_address_space_independent_of_vram, assert_nproc_never_set,
    diagnose_probe_failure, limits_harness_applies, nproc_is_never_set,
    untested_limits,
)
from epic_u2o.safe_harness import ResourceLimits  # noqa: E402


class TestNprocInvariant(unittest.TestCase):
    def test_default_is_zero(self):
        self.assertEqual(ResourceLimits().processes_limit, 0)

    def test_predicate_true_by_default(self):
        self.assertTrue(nproc_is_never_set(ResourceLimits()))

    def test_nproc_32_is_refused(self):
        # the value that produced cudaErrorOsOperationNotSupported, ts
        # 20260927t123630-05fd9bd0
        with self.assertRaises(RlimitReintroductionError) as cm:
            assert_nproc_never_set(ResourceLimits(processes_limit=32))
        self.assertIn("cgroup", str(cm.exception))

    def test_nproc_512_is_refused(self):
        with self.assertRaises(RlimitReintroductionError):
            assert_nproc_never_set(ResourceLimits(processes_limit=512))

    def test_even_generous_2048_is_refused(self):
        # 2048 LOOKS safe and is already fatal: the kernel compares against the
        # real uid's total task count, bracketed between 2048 (aborts) and 3072
        # (works). "Looks generous" is not evidence.
        with self.assertRaises(RlimitReintroductionError):
            assert_nproc_never_set(ResourceLimits(processes_limit=2048))

    def test_apply_never_reaches_setrlimit_nproc(self):
        """The strongest form: exercise apply() for real in a child process.

        apply() sets RLIMIT_AS/DATA/CORE irreversibly, so it cannot run in the
        test process. A child sets processes_limit=1 and must be refused with
        the gate's message, not die from an NPROC abort -- which is what the
        old guarded call would have done, indistinguishably from CUDA failing.
        """
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "from epic_u2o.safe_harness import ResourceLimits\n"
            "ResourceLimits(processes_limit=1).apply()\n"
            "print('APPLIED-WITHOUT-ERROR')\n" % str(REPO)
        )
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("APPLIED-WITHOUT-ERROR", r.stdout)
        self.assertIn("RLIMIT_NPROC is not settable", r.stderr)

    def test_source_contains_no_unguarded_nproc_call(self):
        """Belt and braces: apply() must not set NPROC by any route.

        A guard can be reintroduced by editing this predicate's caller, so the
        literal call is also checked for in the harness source.
        """
        src = (REPO / "epic_u2o" / "safe_harness.py").read_text()
        self.assertNotIn("setrlimit(resource.RLIMIT_NPROC", src)
        self.assertNotIn("setrlimit(resource.RLIMIT_NPROC ", src)


class TestAddressSpaceInvariant(unittest.TestCase):
    def test_as_exceeds_bar1(self):
        self.assertGreater(ResourceLimits().address_space_limit_mb, 32768)

    def test_as_is_not_the_vram_limit(self):
        l = ResourceLimits()
        self.assertNotEqual(l.address_space_limit_mb, l.vram_limit_mb)

    def test_as_24gib_is_refused(self):
        # the value that killed every real-GPU run in ~2 ms
        with self.assertRaises(RlimitReintroductionError) as cm:
            assert_address_space_independent_of_vram(
                ResourceLimits(address_space_limit_mb=24576))
        self.assertIn("BAR1", str(cm.exception))

    def test_coupling_as_to_vram_is_refused(self):
        """The independence must be executable, not a comment.

        If someone 'fixes' the VRAM cap by pointing AS at it, the numbers match
        and the check fires even when the value is large.
        """
        l = ResourceLimits()
        with self.assertRaises(RlimitReintroductionError):
            assert_address_space_independent_of_vram(
                ResourceLimits(address_space_limit_mb=l.vram_limit_mb))


class TestLimitTable(unittest.TestCase):
    def test_every_applied_limit_has_evidence(self):
        self.assertEqual(untested_limits(), [],
                         "an applied limit with no ts job proving it is CUDA-safe")

    def test_every_limit_has_a_reason(self):
        for e in LIMIT_TABLE:
            self.assertTrue(e["why_cuda_safe"].strip(), e["limit"])

    def test_table_matches_what_apply_actually_sets(self):
        """The table must describe the running code, not a different file."""
        applied = limits_harness_applies(ResourceLimits())
        for e in LIMIT_TABLE:
            if e.get("applied", True) and e["limit"] in applied:
                self.assertEqual(
                    applied[e["limit"]] // (1024 * 1024)
                    if e["limit"] in ("RLIMIT_AS", "RLIMIT_DATA")
                    else applied[e["limit"]],
                    e["default"], e["limit"])

    def test_fsize_discrepancy_is_recorded_not_faked(self):
        fsize = next(e for e in LIMIT_TABLE if e["limit"] == "RLIMIT_FSIZE")
        self.assertFalse(fsize["applied"])
        src = (REPO / "epic_u2o" / "safe_harness.py").read_text()
        self.assertNotIn("RLIMIT_FSIZE", src,
                         "FSIZE is now applied; update the table entry")


class TestPreflightWiring(unittest.TestCase):
    """Criterion 3, at the call site a user actually reaches."""

    def test_real_harness_passes(self):
        from epic_u2o.safe_harness import SafeHarness
        ok, msg = SafeHarness().check_rlimit_cuda_compatibility()
        self.assertTrue(ok, msg)
        self.assertIn("RLIMIT_AS", msg)

    def test_injected_bad_limit_is_named(self):
        from epic_u2o.safe_harness import SafeHarness
        bad = ResourceLimits(processes_limit=32)
        ok, msg = SafeHarness().check_rlimit_cuda_compatibility(bad)
        self.assertFalse(ok)
        self.assertIn("rlimit incompatible with CUDA", msg)
        self.assertIn("RLIMIT_NPROC is not settable", msg)

    def test_injected_bad_as_is_named(self):
        from epic_u2o.safe_harness import SafeHarness
        bad = ResourceLimits(address_space_limit_mb=24576)
        ok, msg = SafeHarness().check_rlimit_cuda_compatibility(bad)
        self.assertFalse(ok)
        self.assertIn("BAR1", msg)

    def test_vram_limit_alone_does_not_trip_the_gate(self):
        """A negative control, and the reason the injection parameter exists.

        Lowering the VRAM limit must NOT fail the gate: the invariant is that
        RLIMIT_AS is not DERIVED from vram_limit_mb, so changing one must leave
        the other alone. A gate that fired here would be the (b) bug, not a
        stricter gate.
        """
        from epic_u2o.safe_harness import SafeHarness
        h = SafeHarness(vram_limit_mb=2048)
        ok, msg = h.check_rlimit_cuda_compatibility()
        self.assertTrue(ok, msg)
        self.assertIn("274877906944", msg)   # AS unchanged at 256 GiB


class TestDiagnostic(unittest.TestCase):
    """Criterion 3, injected failure, no GPU."""

    def test_malloc_stage_names_a_limit(self):
        msg = diagnose_probe_failure(
            "malloc", "cudaErrorOsOperationNotSupported",
            {"RLIMIT_AS": 274877906944, "RLIMIT_NPROC": 32})
        self.assertIn("incompatible with CUDA", msg)
        self.assertIn("RLIMIT", msg)

    def test_unknown_stage_says_no_limit_explains_it(self):
        """'Cannot diagnose' must not become 'add a limit'."""
        msg = diagnose_probe_failure("mystery", "someError", {})
        self.assertIn("No limit in the harness's set explains", msg)
        self.assertIn("do not add a limit", msg)

    def test_every_probe_stage_is_mapped(self):
        for stage in ("context", "props", "malloc", "launch", "memcpy"):
            self.assertNotIn("No limit in the harness's set explains",
                             diagnose_probe_failure(stage, "e", {"RLIMIT_AS": 1}))


class TestProbeSource(unittest.TestCase):
    def test_probe_does_the_three_steps_that_actually_broke(self):
        for needle in ("cudaFree(0)", "cudaMalloc", "cudaDeviceSynchronize"):
            self.assertIn(needle, PROBE_SOURCE)

    def test_probe_records_uid_task_count(self):
        self.assertIn("count_uid_tasks", PROBE_SOURCE)
        self.assertIn("uid_tasks", PROBE_SOURCE)

    def test_probe_is_cheap(self):
        # a batch job runs this every time; a 1M-element launch is ~ms
        self.assertIn("1 << 20", PROBE_SOURCE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
