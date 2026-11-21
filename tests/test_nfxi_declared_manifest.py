"""gpu-sim-nfxi: the declared-stage list must cover every stage the gate actually runs.

Main be204ac1 went red with the FULL gate -- "MANIFEST: count mismatch: 14 ran, 13 declared",
on a clean tree -- because scripts/gate.sh carried THREE handwritten copies of the hard-stage
list and gpu-sim-0fr1 added a new stage to two of them.  --fast was green; the mode that runs
on main was red.  The fix is one _DECLARED_ALL, with the mode deriving its list from it.

That fix is only worth anything if it is checked, so this module checks the PROPERTY rather
than a count: every `run_hard "<name>"` site in gate.sh must appear in _DECLARED_ALL, and
_DECLARED_ALL must not name a stage that does not run.  Both directions matter -- a missing
name is the be204ac1 bug, and a stale name is the same bug pointing the other way.

The list is read by letting BASH evaluate the array literal (not by regex over the file), so a
reformatting of the array does not silently turn the check into a text scan (gpu-sim-3qu3).
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "gate.sh"


def _declared_all():
    """Evaluate _DECLARED_ALL by sourcing only that assignment in bash."""
    script = GATE.read_text()
    m = re.search(r"^_DECLARED_ALL=\((.*?)\)\s*$", script, re.S | re.M)
    assert m, "_DECLARED_ALL assignment not found in gate.sh"
    literal = m.group(1)
    r = subprocess.run(
        ["bash", "-c", 'eval "arr=($1)"; printf "%s\\n" "${arr[@]}"', "_", literal],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"bash could not evaluate _DECLARED_ALL: {r.stderr}"
    return [line for line in r.stdout.splitlines() if line.strip()]


def _run_hard_names():
    script = GATE.read_text()
    return re.findall(r'^\s*run_hard\s+"([^"]+)"', script, re.M)


def test_declared_covers_every_run_hard_site():
    declared = _declared_all()
    ran = _run_hard_names()
    assert ran, "no run_hard sites found -- the extraction is broken, not the gate"
    missing = sorted(set(ran) - set(declared))
    stale = sorted(set(declared) - set(ran))
    assert not missing, f"stages run but not declared (the be204ac1 bug): {missing}"
    assert not stale, f"stages declared but never run: {stale}"
    assert len(ran) == len(set(ran)), f"duplicate run_hard site: {ran}"


def test_the_list_is_written_once_and_derived():
    """A second handwritten copy is how two modes came to disagree."""
    script = GATE.read_text()
    # exactly one _DECLARED_ALL literal, and the per-mode list is derived from it
    assert script.count("_DECLARED_ALL=(") == 1, "more than one _DECLARED_ALL assignment"
    assert "DECLARED=()" in script, "the per-mode list is not initialised empty (not derived)"
    assert 'DECLARED+=("$_stage")' in script, "the per-mode list is not built from _DECLARED_ALL"
