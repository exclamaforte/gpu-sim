MUTATIONS = {"e7yw_trunk_resolution": "e7yw trunk resolution prefers main (gpu-sim-e7yw)"}

def apply(root):
    """Mutation for e7yw_trunk_resolution: make resolve_trunk return None."""
    import os
    path = os.path.join(root, "scripts", "resolve_trunk.py")
    backup = open(path).read()
    mutated = backup.replace('            return ref', '            return None  # mutated', 1)
    if mutated == backup:
        raise RuntimeError("e7yw mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path


