"""Mutation for un4y_fail_open: revert fail-closed skip counting to fail-open."""
import os

MUTATIONS = {"un4y_fail_open": "un4y measure.py fail-open on missing binaries (gpu-sim-un4y)"}

def apply(root):
    path = os.path.join(root, "experiments", "gpu-sim-8zqr", "measure.py")
    backup = open(path).read()
    old = "                n_skipped += 1\n                continue"
    new = "                continue"
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("un4y mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
