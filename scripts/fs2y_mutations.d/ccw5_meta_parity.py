def apply(root):
    """Mutation for ccw5_meta_parity: remove a key from META in run.sh."""
    import os
    path = os.path.join(root, "experiments", "gpu-sim-5uu9", "run.sh")
    backup = open(path).read()
    mutated = backup.replace('META = ("loadavg_1_5_15", "foreign_gpu_job", "gate")',
                            'META = ("loadavg_1_5_15", "foreign_gpu_job")', 1)
    if mutated == backup:
        raise RuntimeError("ccw5 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


