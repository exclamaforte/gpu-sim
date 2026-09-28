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
    assert_smoke_binary_is_pure, diagnose_probe_failure,
    limits_harness_applies, nproc_is_never_set, smoke_binary_is_pure,
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


class TestSmokePurity(unittest.TestCase):
    """The PI's MANDATED assertion, verbatim from the 2026-09-27 correction:

    "the purity assertion belongs in this bead's part (1) -- 'ldd <smoke exe>'
    must match neither cudart nor libcuda, checked automatically, so the probe
    in part (2) can never be 'GPU-free' in name only."

    Load-bearing HERE specifically: this probe is about RLIMITS. If the smoke
    path reverts to nvcc's default -cudart=static -- the 6.1.1 defect, where the
    preflight smoke compile ran on the real 5090 -- the probe would still report
    PASS, because nothing about an rlimit probe notices that it executed on the
    device.
    """

    PURE = ("\tlinux-vdso.so.1 (0x00007ffd)\n"
            "\tlibfakegpu.so => /repo/libfakegpu.so (0x00007f00)\n"
            "\tlibc.so.6 => /lib/libc.so.6 (0x00007e00)\n"
            "\tlibm.so.6 => /lib/libm.so.6 (0x00007e00)\n")

    def test_pure_binary_passes(self):
        assert_smoke_binary_is_pure(self.PURE)      # must not raise
        self.assertTrue(smoke_binary_is_pure(self.PURE))

    def test_static_cudart_link_reports_no_libraries(self):
        """THE LIMITATION OF THIS CHECK, stated rather than papered over.

        The exact 6.1.1 defect was a -cudart=STATIC link, and a statically
        linked cudart leaves NO trace in ldd: the symbols are inside the
        executable and the dynamic listing is nearly empty. So this check
        detects the DYNAMIC case and CANNOT, by itself, detect the static one.

        That is why the fix mandated alongside it is '-cudart=none' at COMPILE
        time (gpu-sim-u2o.6.1.3), not a runtime inspection. This test records
        the boundary so nobody later reads a green ldd as proof of purity when
        the dangerous link mode is the one that leaves no evidence.
        """
        static_ldd = ("\tnot a dynamic executable\n"
                      "\tstatically linked, no shared library dependencies\n")
        # the honest result: ldd shows nothing, so this check has nothing to go on
        self.assertTrue(smoke_binary_is_pure(static_ldd))
        self.assertNotIn("libcudart", static_ldd)

    def test_dynamic_cudart_is_rejected(self):
        ldd_out = self.PURE + "\tlibcudart.so.12 => /usr/lib/libcudart.so.12 (0x00007d00)\n"
        self.assertFalse(smoke_binary_is_pure(ldd_out))
        with self.assertRaises(RlimitReintroductionError) as cm:
            assert_smoke_binary_is_pure(ldd_out)
        self.assertIn("NOT GPU-free", str(cm.exception))
        self.assertIn("libcudart", str(cm.exception))

    def test_libcuda_is_rejected(self):
        ldd_out = self.PURE + "\tlibcuda.so.1 => /usr/lib/libcuda.so.1 (0x00007c00)\n"
        self.assertFalse(smoke_binary_is_pure(ldd_out))
        with self.assertRaises(RlimitReintroductionError):
            assert_smoke_binary_is_pure(ldd_out)

    def test_nvml_is_rejected(self):
        """libnvidia-ml is the management path to the same device."""
        self.assertFalse(smoke_binary_is_pure(
            self.PURE + "\tlibnvidia-ml.so.1 => /usr/lib/libnvidia-ml.so.1\n"))

    def test_error_names_the_fix(self):
        with self.assertRaises(RlimitReintroductionError) as cm:
            assert_smoke_binary_is_pure("libcuda.so.1 => /usr/lib/libcuda.so.1")
        self.assertIn("-cudart=none", str(cm.exception))

    def test_check_exists_in_this_bead_not_only_in_6_1_3(self):
        """The PI put the assertion in THIS bead, not only in 6.1.3's shell test.

        A reviewer's item was that neither this test file nor the gate module
        mentioned purity at all, while the capability sat in
        tests/test_cudart_none.sh. Asserted so the wiring cannot be dropped.
        """
        gate = (REPO / "epic_u2o" / "rlimit_cuda_gate.py").read_text()
        self.assertIn("purity", gate)
        self.assertIn("ldd", gate)


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




class TestHarnessDocGenerator(unittest.TestCase):
    """harness.md is GENERATED, and two bugs in that generator are worth freezing.

    Both earlier versions destroyed content or grew the file on every run, and
    both looked fine on the run that produced them.
    """

    RENDERER = REPO / "experiments" / "gpu-sim-u2o.6.3.1" / "render_harness_md.py"
    TARGET = REPO / "epic_u2o" / "harness.md"

    def test_table_lands_beside_the_code_it_describes(self):
        """Not a root harness.md: epic_u2o/safe_harness.py is what it describes."""
        self.assertIn('REPO / "epic_u2o" / "harness.md"', self.RENDERER.read_text())
        self.assertFalse((REPO / "harness.md").exists(),
                         "a root harness.md would be a SECOND harness document")

    def test_generated_region_is_delimited(self):
        txt = self.TARGET.read_text()
        self.assertIn("<!-- BEGIN GENERATED: gpu-sim-u2o.6.3.1 rlimit table -->", txt)
        self.assertIn("<!-- END GENERATED: gpu-sim-u2o.6.3.1 rlimit table -->", txt)

    def test_generator_is_idempotent(self):
        """The bug that bit: run twice, the file GREW.

        The second version split on headers and re-appended its own output, so
        every run duplicated its generated sections. You cannot tell a stale
        file from a correct one when the generator is not a fixed point.
        """
        import subprocess
        before = self.TARGET.read_text()
        r1 = subprocess.run([sys.executable, str(self.RENDERER)],
                            cwd=REPO, capture_output=True, text=True)
        after1 = self.TARGET.read_text()
        subprocess.run([sys.executable, str(self.RENDERER)],
                       cwd=REPO, capture_output=True, text=True)
        after2 = self.TARGET.read_text()
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertEqual(after1, after2, "generator is not a fixed point")
        self.assertEqual(before, after1, "re-running changed a committed artifact")

    def test_generator_preserves_other_beads_sections(self):
        """6.1.2's containment ladder and 6.11's note must survive verbatim."""
        txt = self.TARGET.read_text()
        self.assertIn("cgroup v2", txt)
        self.assertIn("only safe value", txt)
        self.assertIn("VOLATILE", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
