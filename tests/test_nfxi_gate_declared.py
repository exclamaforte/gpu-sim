"""gpu-sim-nfxi (cause 1): the gate's declared-stage list must be DERIVED, not copied per mode.

main went red with "MANIFEST: count mismatch: 14 ran, 13 declared" because the declared list was
written three times and gpu-sim-0fr1 added a stage to two of them: --fast was green while the mode
that runs on main was red. The property is ONE list that is exactly the set of stages the gate runs,
so --fast derives by dropping the suite and nothing else.

Deliberately does not mention any checker module by name: the ledger refuses rows whose bound module
names the checker (see gpu-sim-t108), and this test must be bindable as the bead's liveness row.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "gate.sh"
SUITE = "pytest tests/"


def _declared_and_ran(text):
    m = re.search(r"_DECLARED_ALL=\((.*?)\)\n", text, re.S)
    assert m is not None, (
        "gate.sh must keep hard stage names in a single _DECLARED_ALL list; a per-mode copy is "
        "how gpu-sim-0fr1 made --fast green while the full gate went red")
    declared = re.findall(r'"([^"]+)"', m.group(1))
    ran = re.findall(r'^\s*run_hard "([^"]+)"', text, re.M)
    return declared, ran




import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path



def test_declared_list_covers_every_run_hard_stage():
    declared, ran = _declared_and_ran(GATE.read_text())
    # parse floors: a truncated or empty parse must never pass as "no mismatch"
    assert len(declared) >= 12, f"parsed only {len(declared)} declared stages: {declared}"
    assert len(ran) >= 12, f"parsed only {len(ran)} run_hard stages: {ran}"
    assert SUITE in ran, "the suite stage must be declared and run in full mode"

    # the property: one list, no duplicates, exactly the stages that run
    assert len(set(declared)) == len(declared), f"duplicate stage names: {declared}"
    assert set(declared) == set(ran), (
        "declared and run stages disagree -- "
        f"run but not declared: {sorted(set(ran) - set(declared))}; "
        f"declared but not run: {sorted(set(declared) - set(ran))}")

    # --fast derives by dropping the suite and NOTHING else
    fast = [s for s in declared if s != SUITE]
    assert set(fast) == set(ran) - {SUITE}, fast

    # the mechanism is live: dropping ONE declared name must be reported, not tolerated
    sys.path.insert(0, str(ROOT / "scripts"))
    import gate_manifest
    assert gate_manifest.compare(declared, ran) == ([], [], []), "clean list must compare clean"
    missing, undeclared, _dupes = gate_manifest.compare(declared[1:], ran)
    assert undeclared and not missing, (
        "gate_manifest did not report a stage that ran but is no longer declared "
        f"(missing={missing}, undeclared={undeclared})")


