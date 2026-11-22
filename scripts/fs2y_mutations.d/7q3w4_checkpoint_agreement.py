"""Mutation for 7q3w4_checkpoint_agreement (gpu-sim-7q3w.4).

Makes the checkpoint equality check pass unconditionally, the exact defect a
hash-equality checker must not have.  The bound test's synthetic disagreement
must then fail; its equal-hash arm still passes, so the mutation specifically
blinds the disagreement predicate rather than breaking the hash function.
"""
import os

MUTATIONS = {
    "7q3w4_checkpoint_agreement":
    "7q3w.4: make the checkpoint agreement check pass unconditionally"
}


def apply(root):
    path = os.path.join(
        root, "experiments", "gpu-sim-7q3w.4", "checkpoint_agreement.py"
    )
    backup = open(path).read()
    anchor = "    agrees = len(distinct) == 1\n"
    # Longer than the original, so a stale .pyc cannot survive the restore on a
    # same-second mtime (the trap recorded in 3v6d1_fabric_rate).
    mutated = backup.replace(
        anchor,
        "    agrees = True  # MUTATION: unconditional pass blinds disagreement\n",
        1,
    )
    if mutated == backup:
        raise RuntimeError(
            "7q3w4_checkpoint_agreement: agreement anchor not found"
        )
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
