"""gpu-sim-vuoq: the plant-verification contract, tested on REAL pytest output.

The defect: "the bound test failed" was read as "the plant bit", so a mutation
that broke the target's SYNTAX certified coverage (gpu-sim-f1sl: one line of a
four-line pytest.fail(...) replaced, leaving an IndentationError; the checker
returned rc=0).  A plant must leave the target PARSEABLE and must fail as a test
outcome.

This is a product test: it imports scripts/plant_verify.py, not the checker, so
a ledger row can bind it.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _collect_only(target):
    import subprocess
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(target)],
        capture_output=True, text=True, cwd=str(REPO))


def test_broken_module_is_unparseable(tmp_path):
    import plant_verify

    broken = tmp_path / "test_broken_fixture.py"
    broken.write_text("def test_ok():\n    assert 1 + 1 == 2\n        dangling\n")
    run = _collect_only(broken)
    assert run.returncode != 0
    msg = plant_verify.collection_failure(run)
    assert msg.startswith("IndentationError at "), msg
    assert f"test_broken_fixture.py:3" in msg, msg

    good = tmp_path / "test_good_fixture.py"
    good.write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    run_ok = _collect_only(good)
    assert run_ok.returncode == 0
    assert plant_verify.collection_failure(run_ok) == "", "a healthy module must not be refused"


def test_assertion_failure_is_not_a_collection_error(tmp_path):
    """The fix must not refuse the plant it is meant to accept: a real assertion
    failure is a test outcome, not a collection error."""
    import plant_verify
    import subprocess

    t = tmp_path / "test_failing_fixture.py"
    t.write_text("def test_ok():\n    assert 1 + 1 == 3\n")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", str(t)],
                         capture_output=True, text=True, cwd=str(REPO))
    assert run.returncode == 1
    assert plant_verify.collection_failure(run) == ""
    assert plant_verify.non_assertion_failure(run) == ""


def test_mutation_diff_reports_the_replaced_span(tmp_path):
    import plant_verify
    before = "def f():\n    raise SystemExit(1)\n    return 2\n"
    p = tmp_path / "m.py"
    p.write_text(before)
    p.write_text("def f():\n    raise SystemExit(1)\n        return 2\n")
    diff = plant_verify.mutation_diff(before, str(p))
    assert "+++ after" in diff and "return 2" in diff, diff
