"""Mutation for c3hd_live_owner_classifier (gpu-sim-c3hd).

Blinds the safety property the whole reaper rests on: "delete only what no live
process owns". The liveness check is made to answer "not owned" unconditionally --
the shape cywp used ("if False and ...") -- so every live-owned scratch becomes
deletable. Reviewer's requirement: the mutation must blind THAT rule, not a
peripheral one.
"""
import os

MUTATIONS = {"c3hd_live_owner_classifier":
             "c3hd: make the live-owner classifier answer 'not owned' unconditionally"}


def apply(root):
    path = os.path.join(root, "scripts", "reap_scratch_labs.py")
    backup = open(path).read()
    anchor = "    cwds, cmdlines = owners\n    s = str(path)"
    if anchor not in backup:
        raise RuntimeError("c3hd_live_owner_classifier: owners unpack not found")
    # "not owned" unconditionally: every scratch becomes a candidate, which is the
    # failure the bead exists to prevent (a live test's scratch deleted mid-run)
    mutated = backup.replace(anchor, "    return False  # MUTATION: nobody owns anything\n" + anchor, 1)
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
