"""gpu-sim-vuoq: a syntax-breaking mutation must not certify coverage.

The defect: the plant verification read "the bound test failed" as "the plant
bit", so a mutation that broke the target's SYNTAX (one line of a four-line
pytest.fail(...) replaced, leaving dangling continuation lines) made pytest exit
nonzero and was recorded as covering the rule.  On gpu-sim-f1sl that was
`E IndentationError: unexpected indent` / `ERROR tests/test_verify_committed_run.py`
and the checker returned rc=0.

These tests pin the two halves of the fix: the classifier reads a REAL broken
module (not a hand-written string), and the checker, run end to end against a
scratch repo with a syntax-breaking registered mutation, exits NONZERO naming
the file and the error.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "check_append_only.py"


def _pytest_collect_only(target):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(target)],
        capture_output=True, text=True, cwd=str(REPO))


def _load_checker():
    sys.path.insert(0, str(REPO / "scripts"))
    import check_append_only  # noqa: E402
    return check_append_only


def test_broken_module_is_unparseable():
    """collection_failure() names the error and the file:line for a real
    IndentationError, and does not fire on a healthy module."""
    checker = _load_checker()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        broken = Path(td) / "test_broken_fixture.py"
        broken.write_text("def test_ok():\n    pass\n        dangling\n")
        run = _pytest_collect_only(broken)
        assert run.returncode != 0
        msg = checker.collection_failure(run)
        assert msg.startswith("IndentationError at "), msg
        assert "test_broken_fixture.py:3" in msg, msg

        good = Path(td) / "test_good_fixture.py"
        good.write_text("def test_ok():\n    pass\n")
        run_ok = _pytest_collect_only(good)
        assert run_ok.returncode == 0
        assert checker.collection_failure(run_ok) == ""


@pytest.fixture()
def scratch_repo(tmp_path):
    """A minimal repo the checker can replay one row in."""
    root = tmp_path / "repo"
    (root / "scripts" / "fs2y_mutations.d").mkdir(parents=True)
    (root / "tests" / "fs2y_ledger.d").mkdir(parents=True)
    for py in list(CHECKER.parent.glob("*.py")):
        shutil.copyfile(py, root / "scripts" / py.name)
    (root / "tests" / "test_vuoq_fixture_target.py").write_text(
        "def test_bound():\n    assert 1 + 1 == 2\n")
    (root / "tests" / "fs2y_ledger.d" / "vuoq_fixture.json").write_text(json.dumps({
        "name": "vuoq_fixture",
        "beads": ["gpu-sim-vuoq"],
        "state": "passing",
        "test": "tests/test_vuoq_fixture_target.py::test_bound",
        "bound_calls": 1,
        "mutation": "vuoq_syntax_break",
        "verification": "mutation_plant",
    }, indent=2) + "\n")
    (root / "scripts" / "fs2y_mutations.d" / "vuoq_syntax_break.json").write_text(json.dumps({
        "key": "vuoq_syntax_break",
        "description": "gpu-sim-vuoq fixture: break the bound test's syntax",
        "function": "vuoq_syntax_break",
    }, indent=2) + "\n")
    (root / "scripts" / "fs2y_mutations.d" / "vuoq_syntax_break.py").write_text(
        'from pathlib import Path\n\n'
        'PATH = "tests/test_vuoq_fixture_target.py"\n\n'
        'def apply(root):\n'
        '    p = Path(root) / PATH\n'
        '    backup = p.read_text()\n'
        '    # the f1sl shape: an anchor that replaced one line of a multi-line\n'
        '    # statement, leaving a dangling continuation line\n'
        '    p.write_text("def test_bound():\\n    assert 1 + 1 == 2\\n        dangling\\n")\n'
        '    return backup, str(p)\n')
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True, env=env)
    return root


def test_checker_refuses_syntax_breaking_plant(scratch_repo):
    """THE acceptance case: the mutation leaves the target unparseable, so the
    checker must exit nonzero and say so -- today it returned rc=0."""
    run = subprocess.run(
        [sys.executable, str(scratch_repo / "scripts" / "check_append_only.py"),
         "--rows", "vuoq_fixture", "--dirty-ok", "--force"],
        cwd=str(scratch_repo), capture_output=True, text=True, timeout=600)
    out = run.stdout + run.stderr
    assert run.returncode != 0, out[-3000:]
    assert "mutation left the test module unparseable" in out, out[-3000:]
    assert "IndentationError at " in out, out[-3000:]
    assert "mutation span" in out, out[-3000:]
    # the tree must be restored even when the plant is refused
    dirty = subprocess.run(["git", "-C", str(scratch_repo), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    assert dirty == "", dirty
