MUTATIONS = {"mve4_trunk_resolution": "mve4 hermetic gate: resolve_trunk prefers main (gpu-sim-mve4)"}

def apply(root):
    import os
    path = os.path.join(root, "scripts", "resolve_trunk.py")
    backup = open(path).read()
    mutated = backup.replace('            return ref', '            return None  # mutated', 1)
    if mutated == backup:
        raise RuntimeError("mve4 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
