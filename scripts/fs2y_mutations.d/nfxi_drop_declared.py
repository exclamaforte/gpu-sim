"""gpu-sim-nfxi: drop one stage name from _DECLARED_ALL -- the exact be204ac1 bug.

Main be204ac1 was red because a stage ran and was not declared.  The fix is one list, and the
bound test asserts the PROPERTY (every run_hard site is declared, and no declared name is
unrun).  This applier reproduces the defect in the copy it is handed: it removes a real stage
name from the single list, leaving the run_hard site in place, so the property is broken
without breaking the file -- the script still parses, the gate still runs, and only a
property check can see it.
"""

import os

MUTATIONS = {
    "nfxi_drop_declared": "nfxi: drop a stage name from the single declared list (the be204ac1 bug)",
}

PATH = ("scripts", "gate.sh")
VICTIM = '"trunk refs agree" "shard format")'
BROKEN = '"trunk refs agree")'


def apply(root):
    path = os.path.join(root, *PATH)
    backup = open(path).read()
    if VICTIM not in backup:
        raise RuntimeError(
            "nfxi_drop_declared: the declared-list line was rewritten; this mutation no longer "
            "describes the code")
    mutated = backup.replace(VICTIM, BROKEN, 1)
    if mutated == backup:
        raise RuntimeError("nfxi_drop_declared mutation did not change the file")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
