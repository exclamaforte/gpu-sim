"""Mutation for cywp_headroom_detect (gpu-sim-cywp).

Makes the btrfs METADATA headroom detector always report healthy: the gate stage
that exists because rename() failed ENOSPC while `df` said 70 GiB free would then
pass on the exact state it was written for. The bound test feeds it the measured
outage numbers (98% used, 1 MiB unallocated) and must fail by assertion.
"""
import os

MUTATIONS = {"cywp_headroom_detect":
             "cywp: make the fs-metadata headroom detector always report healthy"}


def apply(root):
    path = os.path.join(root, "scripts", "check_fs_headroom.py")
    backup = open(path).read()
    anchor = "    if full or warm:\n"
    mutated = backup.replace(anchor, "    if False and (full or warm):  # MUTATION\n", 1)
    if mutated == backup:
        raise RuntimeError("cywp_headroom_detect: failure branch anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
