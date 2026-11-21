MUTATIONS = {"fu0h_reap": "fu0h reaper removes dead-PID build dirs (gpu-sim-fu0h)"}

def apply(root):
    import os
    path = os.path.join(root, "experiments", "gpu-sim-7q3w.1", "run.sh")
    backup = open(path).read()
    start = backup.index("# fu0h-reaper-start")
    end = backup.index("# fu0h-reaper-end") + len("# fu0h-reaper-end")
    mutated = backup[:start] + backup[end:]
    if mutated == backup:
        raise RuntimeError("fu0h mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
