MUTATIONS = {"njh1_body_exception": "gpu-sim-njh1: accept a test-body exception as a plant again (drop the exception classifier)"}

"""Mutation for njh1_body_exception (gpu-sim-njh1): make the body-exception
classifier unreachable, so a test that raises NameError is credited again."""

from pathlib import Path

PATH = "scripts/plant_verify.py"
ANCHOR = "    if m2:\n"

def apply(root):
    p = Path(root) / PATH
    backup = p.read_text()
    if ANCHOR not in backup:
        raise SystemExit("njh1_body_exception: classifier anchor not found")
    p.write_text(backup.replace(ANCHOR, "    if False:  # MUTATION (njh1)\n", 1))
    return backup, str(p)
