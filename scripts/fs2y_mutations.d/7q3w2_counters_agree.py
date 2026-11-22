"""Mutation for 7q3w2_counters_agree (gpu-sim-7q3w.2).

Makes the printed-counters/file agreement check IGNORE the field that matters
most -- isend, the number the original defect got wrong -- so a log that says the
data path was never used compares "clean". The bound test must then fail by
assertion against its doctored fixture.
"""
import os

MUTATIONS = {"7q3w2_counters_agree":
             "7q3w.2: make the counters agreement check skip isend"}


def apply(root):
    path = os.path.join(root, "experiments", "gpu-sim-7q3w.2", "check_counters.py")
    backup = open(path).read()
    anchor = "    for pname, fname in NAME_MAP.items():\n"
    mutated = backup.replace(
        anchor,
        anchor + "        if pname == \"isend\":\n            continue  # MUTATION\n",
        1)
    if mutated == backup:
        raise RuntimeError("7q3w2_counters_agree: comparison loop anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
