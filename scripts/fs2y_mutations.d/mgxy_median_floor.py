"""Mutation for gpu-sim-mgxy: put the cold median back as the launch floor."""
MUTATIONS = {"mgxy_median_floor":
             "mgxy: launch floor reverts to the median of the whole empty series"}


def apply(root):
    import os
    path = os.path.join(root, "experiments", "gpu-sim-8zqr", "measure.py")
    backup = open(path).read()
    old = "    w = warm(empty_trials)\n    return min(w) if w else 0.0\n"
    new = "    return med(empty_trials)\n"
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("mgxy mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
