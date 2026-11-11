MUTATIONS = {"7m60_nonpositive_kernel": "7m60 measure.py non-positive kernel rejection reason (gpu-sim-7m60)"}

def apply(root):
    """Mutation for 7m60_nonpositive_kernel: revert reason string to old form."""
    import os
    path = os.path.join(root, "experiments", "gpu-sim-8zqr", "measure.py")
    backup = open(path).read()
    old = '"non_positive_kernel"'
    new = '"nonpositive_kernel_us"'
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("7m60 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
