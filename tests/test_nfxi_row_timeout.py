"""gpu-sim-nfxi (cause 2): a row timeout must never read as a bitten plant.

check_append_only's _run_pytest re-raised subprocess.TimeoutExpired, so a slow machine arrived as a
ledger defect ("ledger test-id resolution: Command '[...]'"); worse, at the mutation-run site rc != 0
is the DESIRED outcome, so a timeout could have been credited as a plant that bit.

NOTE: this module names the checker, so the ledger's self-reference rule refuses any row bound to it
(gpu-sim-t108). The bead's liveness row is therefore bound to the cause-1 test in
tests/test_nfxi_gate_declared.py; this file still guards the property in the suite, and the plants
below prove it bites.
"""

import importlib.util
import subprocess
import tempfile
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_append_only.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("cao_nfxi", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_pytest_reports_124_on_timeout():
    """The conversion itself: a killed run is rc=124 with a named marker, not an exception and
    not a bare failure. This also pins the `out` initialisation (a timeout that fires before the
    assignment used to raise UnboundLocalError)."""
    mod = _load_checker()
    t0 = time.time()
    r = mod._run_pytest([sys.executable, "-c", "import time; time.sleep(30)"],
                        ROOT, 0.05, mod._child_env())
    elapsed = time.time() - t0
    assert r.returncode == 124, f"expected the timeout code 124, got {r.returncode}"
    assert "ROW TIMEOUT" in r.stdout, r.stdout
    assert "infrastructure, not a ledger defect" in r.stdout, r.stdout
    assert elapsed < 10, f"the killed run took {elapsed:.1f}s"


def test_row_timeout_is_not_a_bitten_plant(monkeypatch):
    """The dangerous site: the mutation run, where rc != 0 means the plant BIT. A timeout there
    must report TIMEOUT and must not report that the mutation failed (which would be a plant
    credited by infrastructure)."""
    mod = _load_checker()
    # project-local scratch: /tmp is quota-limited tmpfs (gpu-sim-u6j5)
    scratch = ROOT / ".lab" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    _td = tempfile.TemporaryDirectory(dir=scratch)
    victim = Path(_td.name) / "victim.txt"
    victim.write_text("ORIGINAL\n")

    def fake_apply(root):
        victim.write_text("MUTATED\n")
        return ("ORIGINAL\n", str(victim))

    row = {
        "name": "nfxi_timeout_probe",
        "beads": ["gpu-sim-nfxi"],
        "state": "passing",
        "test": "tests/test_nfxi_gate_declared.py::test_run_pytest_reports_124_on_timeout",
        "bound_calls": 1,
        "mutation": "nfxi_timeout_probe",
        "verification": "mutation_plant",
    }
    monkeypatch.setattr(mod, "load_ledger_entries", lambda root=None: [row])
    monkeypatch.setattr(
        mod, "_load_mutations",
        # (registry, error) where registry is (MUTATIONS, MUTATION_FUNCTIONS)
        lambda root=None: (({row["mutation"]: "probe plant (gpu-sim-nfxi)"},
                            {row["mutation"]: fake_apply}), None))
    # let the collect-only resolution succeed, then drive the run result by hand
    real_run = subprocess.run

    def fake_run(cmd, *a, **kw):
        if isinstance(cmd, list) and "--collect-only" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "1 test collected\n", "")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    seq = [1, 0]

    def fake_pytest(cmd, root, timeout, env):
        return subprocess.CompletedProcess(cmd, seq.pop(0) if seq else 0, "", "")

    monkeypatch.setattr(mod, "_run_pytest", fake_pytest)

    # polarity control: fail then pass is a plant that bit -> no error at all
    errs = [e for e in mod.check_ledger_test_ids(ROOT) if row["name"] in e]
    assert errs == [], f"a normally-biting plant reported errors: {errs}"

    # the danger: 124 at the run that MUST fail
    seq[:] = [124]
    errs = [e for e in mod.check_ledger_test_ids(ROOT) if row["name"] in e]
    assert errs, "a mutation-run timeout produced no error at all"
    assert any("TIMEOUT (infrastructure, not a ledger defect)" in e and "mutation run" in e
               for e in errs), errs
    assert not any("did not fail" in e for e in errs), (
        "a timeout was reported as a mutation that failed (i.e. a plant credited by "
        f"infrastructure): {errs}")
    assert mod._APPLIED_MUTATIONS == [], "the applied mutation was not restored"
    _td.cleanup()
