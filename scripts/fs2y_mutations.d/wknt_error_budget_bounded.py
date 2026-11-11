MUTATIONS = {"wknt_error_budget_bounded": "wknt error budget bounded (gpu-sim-wknt)"}

def apply(root):
    """Mutation for wknt_error_budget_bounded: corrupt the sort key."""
    import os
    path = os.path.join(root, "scripts", "error_budget.py")
    backup = open(path).read()
    mutated = backup.replace('    f = r["fraction_one_sided"]',
                            '    f = r["flip_down"]', 1)
    if mutated == backup:
        raise RuntimeError("wknt mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


