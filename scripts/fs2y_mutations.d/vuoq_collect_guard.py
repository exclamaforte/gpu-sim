"""Mutation for vuoq_collect_guard (gpu-sim-vuoq).

Restores the pre-fix behavior in the product module: the classifier never
reports an unparseable module, so a syntax-breaking mutation is accepted as a
plant that bit.  The bound test feeds it REAL pytest output from a dangling
continuation line and must fail.
"""

from pathlib import Path

PATH = "scripts/plant_verify.py"
ANCHOR = "    text = (run.stdout or \"\") + (run.stderr or \"\")\n    loc = \"\""
REPLACEMENT = ("    return \"\"  # MUTATION: never report an unparseable module (vuoq)\n"
               "    text = (run.stdout or \"\") + (run.stderr or \"\")\n    loc = \"\"")


def apply(root):
    p = Path(root) / PATH
    backup = p.read_text()
    if ANCHOR not in backup:
        raise SystemExit("vuoq_collect_guard: classifier anchor not found")
    p.write_text(backup.replace(ANCHOR, REPLACEMENT, 1))
    return backup, str(p)
