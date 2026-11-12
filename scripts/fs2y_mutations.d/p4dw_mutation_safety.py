MUTATIONS = {"p4dw_mutation_safety": "p4dw mutation restore safety (gpu-sim-p4dw)"}

def apply(root):
    """Mutation for p4dw_mutation_safety: disable the restore safety net."""
    import os
    path = os.path.join(root, "scripts", "check_append_only.py")
    backup = open(path).read()
    mutated = backup.replace("atexit.register(_restore_all_mutations)",
                             "pass  # atexit disabled (p4dw plant)", 1)
    mutated = mutated.replace("signal.signal(_sig, _signal_handler)",
                              "pass  # signal disabled (p4dw plant)", 1)
    if mutated == backup:
        raise RuntimeError("p4dw mutation anchor not found")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
